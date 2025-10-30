import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from backend.model.model_utils import SiameseNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reuse the same preprocessing as training: grayscale, 155x220, normalized
# Match ResNet18 (pretrained) expected 3-channel normalization
_transform = transforms.Compose([
    transforms.Resize((155, 220)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

_model_singleton = None

def _ensure_model_loaded():
    global _model_singleton
    if _model_singleton is None:
        model = SiameseNetwork().to(device)
        # Load the existing checkpoint in the repo
        checkpoint_path = "backend/model/siamese_model.pth"
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        _model_singleton = model
    return _model_singleton

def _to_model_tensor(img_pil: Image.Image):
    # Convert to RGB to match ResNet18 3-channel input
    img = img_pil.convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(device)
    return tensor

def get_similarity_score(img1_pil: Image.Image, img2_pil: Image.Image, threshold: float = 0.85):
    model = _ensure_model_loaded()
    img1 = _to_model_tensor(img1_pil)
    img2 = _to_model_tensor(img2_pil)

    with torch.no_grad():
        emb1 = model.forward_once(img1)
        emb2 = model.forward_once(img2)
        sim_score = F.cosine_similarity(emb1, emb2).item()

    verdict = "Same person" if sim_score >= threshold else "Different person"
    return sim_score, verdict

def compare_signatures(img_path1, img_path2):
    # Helper for CLI/testing with file paths
    model = _ensure_model_loaded()
    img1 = _to_model_tensor(Image.open(img_path1))
    img2 = _to_model_tensor(Image.open(img_path2))
    with torch.no_grad():
        emb1 = model.forward_once(img1)
        emb2 = model.forward_once(img2)
        sim_score = F.cosine_similarity(emb1, emb2).item()
    return sim_score

def generate_saliency_heatmap(img1_pil: Image.Image, img2_pil: Image.Image):
    model = _ensure_model_loaded()
    img1 = _to_model_tensor(img1_pil)
    img2 = _to_model_tensor(img2_pil)

    img2.requires_grad_(True)
    # Normalize to unit vectors to stabilize gradient scale
    with torch.enable_grad():
        emb1 = model.forward_once(img1)
        emb2 = model.forward_once(img2)
        sim = F.cosine_similarity(emb1, emb2)
        # Backpropagate similarity to img2 inputs
        sim.backward(torch.ones_like(sim))
        grad = img2.grad.detach().squeeze(0)  # C x H x W
        sal = grad.abs().sum(dim=0)  # H x W
        sal = sal / (sal.max() + 1e-8)

    sal_np = (sal.cpu().numpy() * 255.0).astype('uint8')
    # Create red heatmap with alpha channel from saliency
    h, w = sal_np.shape
    rgba = Image.new('RGBA', (w, h), (255, 0, 0, 0))
    alpha = Image.fromarray(sal_np, mode='L')
    rgba.putalpha(alpha)
    return rgba

def generate_gradcam_heatmap(img_pil: Image.Image):
    model = _ensure_model_loaded()
    model.zero_grad(set_to_none=True)
    # Access last conv block of resnet18
    target_layer = model.backbone.layer4[-1].conv2
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations['value'] = out.detach()
    def bwd_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    handle_f = target_layer.register_forward_hook(fwd_hook)
    handle_b = target_layer.register_full_backward_hook(bwd_hook)

    try:
        x = _to_model_tensor(img_pil)
        with torch.enable_grad():
            emb = model.forward_once(x)
            # Maximize L2 norm of embedding as a generic target
            target = (emb.pow(2).sum(dim=1)).mean()
            target.backward()

        A = activations['value']  # [B,C,H,W]
        dA = gradients['value']   # [B,C,H,W]
        weights = dA.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
        cam = (weights * A).sum(dim=1, keepdim=False)  # [B,H,W]
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        cam_np = (cam.squeeze(0).cpu().numpy() * 255).astype('uint8')
        h, w = cam_np.shape
        rgba = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        alpha = Image.fromarray(cam_np, mode='L')
        red = Image.new('L', (w, h), 255)
        rgba.putalpha(alpha)
        rgba.putpixel((0,0), (255,0,0,0))  # force RGBA mode
        # Compose red with alpha
        rgba = Image.merge('RGBA', (red, Image.new('L',(w,h),0), Image.new('L',(w,h),0), alpha))
        return rgba
    finally:
        handle_f.remove()
        handle_b.remove()

if __name__ == "__main__":
    sim = compare_signatures("img1.png", "img2.png")
    print("Cosine Similarity:", sim)
    print("✅ Same person" if sim > 0.85 else "❌ Different person")

