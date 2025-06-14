# backend/model/inference.py

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from backend.model.models import SiameseNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path):
    transform = Compose([
        Resize((155, 220)),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])
    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)
    return img

def load_model(checkpoint_path):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def compare_signatures(img_path1, img_path2, model):
    img1 = load_image(img_path1)
    img2 = load_image(img_path2)

    with torch.no_grad():
        emb1 = model.forward_once(img1)
        emb2 = model.forward_once(img2)
        sim_score = F.cosine_similarity(emb1, emb2).item()

    return sim_score

if __name__ == "__main__":
    model = load_model("saved_models/siamese_final.pth")
    sim = compare_signatures("img1.png", "img2.png", model)
    print("Cosine Similarity:", sim)
    print("✅ Same person" if sim > 0.85 else "❌ Different person")

