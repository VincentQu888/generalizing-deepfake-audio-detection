from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from embedding_pipeline.embedder import Embedder
from embedding_pipeline.stft import stft
from detecting_pipeline.memory import DetectorMemory


class Detector:
	def __init__(
     	self, 
		raw_vectors: torch.Tensor,
        labels: list[int],
        embedder: Embedder, 
        cosine_similarity_threshold: float = 0.99, 
        similar_ratio_threshold: float = 0.5,
        device: torch.device | str = "cpu"
    ):	
		embedder.eval()
		self.embedder = embedder
		self.cosine_similarity_threshold = cosine_similarity_threshold
		self.similar_ratio_threshold = similar_ratio_threshold
		self.device = torch.device(device)
		self.memory = DetectorMemory(
			raw_vectors=raw_vectors,
			labels=labels,
			fake_ratio_threshold=similar_ratio_threshold,
			similarity_threshold=cosine_similarity_threshold,
			min_cluster_size=5,
			top_k=100
		)
  
	def update_memory(self, audio_path: str | Path):
		"""
		update memory with new reference vectors from audio file, only for fake vectors
		"""
		magnitude_db, _ = stft(str(audio_path))
		query = torch.tensor(magnitude_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device) # double unsqueeze for channel and batch
		with torch.no_grad():
			query_embeddings = self.embedder(query)
   
		new_vectors = query_embeddings.reshape(-1, query_embeddings.size(-1))
		self.memory.update_memory(new_vectors=new_vectors)

	def detect(self, audio_path: str | Path) -> dict[str, Any]:
		"""
  		detect whether one audio file is deepfake
    	"""
		magnitude_db, _ = stft(str(audio_path))
		query = torch.tensor(magnitude_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device) # double unsqueeze for channel and batch
		with torch.no_grad():
			query_embeddings = self.embedder(query)
		query_vectors = query_embeddings.reshape(-1, query_embeddings.size(-1))

		similar_count, best_similarity = self.count_similar_vectors(
			query_vectors=query_vectors,
			cosine_similarity_threshold=self.cosine_similarity_threshold
		)

		similar_ratio = float(similar_count / max(1, query_vectors.size(0)))
		is_deepfake = similar_ratio >= self.similar_ratio_threshold
		return {
			"num_query_vectors": int(query_vectors.size(0)),
			"similar_count": int(similar_count),
			"similar_ratio": similar_ratio,
			"best_similarity": best_similarity,
			"is_deepfake": bool(is_deepfake),
			"pred_label": 1 if is_deepfake else 0,
		}

	def count_similar_vectors(
     	self,
		query_vectors: torch.Tensor,
		cosine_similarity_threshold: float,
		chunk_size: int = 4096,
	) -> tuple[int, float]:
		"""
  		count query vectors whose aggregated cosine similarity exceeds threshold
		"""

		query = query_vectors.float()
		ref = self.memory.reference_vectors
		ref = torch.tensor(ref).float().to(query.device)

		query = query / query.norm(dim=1, keepdim=True).clamp_min(1e-12)
		ref = ref / ref.norm(dim=1, keepdim=True).clamp_min(1e-12)

		max_sims_per_query: list[torch.Tensor] = []
		total = query.size(0)

		for start in range(0, total, chunk_size):
			end = min(start + chunk_size, total)
			sims = query[start:end] @ ref.T
			scores = sims.mean(dim=1)
			max_sims_per_query.append(scores)

		all_max_sims = torch.cat(max_sims_per_query, dim=0)
		similar_count = int((all_max_sims >= cosine_similarity_threshold).sum().item())
		best_similarity = float(all_max_sims.max().item())
		return similar_count, best_similarity
