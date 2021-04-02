import re
from pathlib import Path
from typing import List

import numpy as np
import requests
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


class AgitationModel(torch.nn.Module):
    MODEL_GDRIVE_ID = '1IqNhXJXOXBVEVB8r0FHXN2LlYIEqY8fd'
    MODEL_PATH = Path(__file__, '..', '..', 'data', 'herbert_mlp.pt').resolve()
    LABELS = ['normal', 'voting_turnout', 'encouragement', 'inducement']

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('allegro/herbert-klej-cased-tokenizer-v1')
        self.transformer_model = AutoModel.from_pretrained('allegro/herbert-klej-cased-v1')
        self.layers = nn.Sequential(
            nn.Linear(self.transformer_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(self.LABELS)),
            nn.Softmax(dim=1),
        )

    @classmethod
    def from_pretrained(cls) -> 'AgitationModel':
        model = cls()
        if not cls.MODEL_PATH.is_file():
            model._download_model()
        model.load_state_dict(torch.load(cls.MODEL_PATH, map_location='cpu'))
        return model

    def forward(self, x: List[str]) -> torch.Tensor:
        tokenized = self.tokenizer(x, padding=True, return_tensors='pt')
        embedding = self.transformer_model(**tokenized).pooler_output
        x = self.layers(embedding)
        return x

    def predict(self, x: List[str]) -> np.ndarray:
        self.eval()
        x = [self._clean_text(text) for text in x]
        with torch.no_grad():
            return np.array(self.LABELS)[self(x).numpy().argmax(axis=1)]

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(
            r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\("
            r"([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`"
            r"!()\[\]{};:'\".,<>?«»“”‘’]))", '', text)
        text = re.sub(r'[^a-ząćęłńóśżź .,?!*#@0-9%]', '', text)
        return text

    def _download_model(self) -> None:
        gdrive_url = "https://docs.google.com/uc?export=download"

        session = requests.Session()
        response = session.get(gdrive_url, params={'id': self.MODEL_GDRIVE_ID}, stream=True)

        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value

        if token:
            params = {'id': self.MODEL_GDRIVE_ID, 'confirm': token}
            response = session.get(gdrive_url, params=params, stream=True)
            total_size = int(response.headers.get('content-length', default=0))

            self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            progress_bar = tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading the model",
            )

            with self.MODEL_PATH.open(mode='wb') as file:
                for chunk in response.iter_content(chunk_size=128):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

            progress_bar.close()
