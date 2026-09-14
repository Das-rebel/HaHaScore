# Push fusion_v2_test.pt to Hugging Face
import torch
import json
import os
from transformers import RobertaModel, WavLMModel, RobertaConfig, WavLMConfig
from huggingface_hub import HfApi, create_repo

REPO_ID = "Hayasuki/chucklenet-fusion-v2"
MODEL_PATH = os.path.expanduser("~/models/chuckle_predictor/fusion_v2_test.pt")

# Define the model architecture matching saved weights
class FusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.txt = RobertaModel.from_pretrained("roberta-base")
        self.aud = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.fuse = torch.nn.Sequential(
            torch.nn.Linear(768 + 512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1))

    def forward(self, ids, mask, wav):
        t = self.txt(ids, attention_mask=mask).pooler_output
        a = self.aud(wav).extract_features.mean(1)
        return self.fuse(torch.cat([t, a], -1)).squeeze(-1)

print("Loading model weights...")
model = FusionModel()
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()
print("Model loaded successfully")

# Create repo if not exists
api = HfApi()
try:
    create_repo(REPO_ID, private=False, exist_ok=True)
    print(f"Repo ready: {REPO_ID}")
except Exception as e:
    print(f"Repo create: {e}")

# Save model + configs locally for push
LOCAL_DIR = "/tmp/fusion_hf"
os.makedirs(LOCAL_DIR, exist_ok=True)

# Save model
torch.save(model.state_dict(), os.path.join(LOCAL_DIR, "pytorch_model.bin"))

# Save configs
model.txt.config.save_pretrained(os.path.join(LOCAL_DIR, "text_encoder"))
model.aud.config.save_pretrained(os.path.join(LOCAL_DIR, "audio_encoder"))

# Save fusion config
import json
fusion_config = {
    "architectures": ["FusionModel"],
    "model_type": "chucklenet_fusion",
    "text_encoder": "roberta-base",
    "audio_encoder": "microsoft/wavlm-base-plus",
    "fusion_hidden": [256, 64],
    "dropout": [0.3, 0.2],
    "num_labels": 1,
    "task": "laughter_prediction_binary"
}
with open(os.path.join(LOCAL_DIR, "config.json"), "w") as f:
    json.dump(fusion_config, f, indent=2)

# Push
print(f"Pushing to {REPO_ID}...")
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    commit_message="Initial fusion model push (roberta-base + wavlm-base-plus)",
    token=os.environ.get("HF_TOKEN")
)
print("Done!")