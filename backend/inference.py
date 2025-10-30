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

if __name__ == "__main__":
    sim = compare_signatures("img1.png", "img2.png")
    print("Cosine Similarity:", sim)
    print("✅ Same person" if sim > 0.85 else "❌ Different person")

