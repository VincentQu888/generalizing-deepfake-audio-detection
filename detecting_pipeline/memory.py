import hnswlib as hnsw
import numpy as np
import torch


class DetectorMemory:
    """
    memory for detector to store reference vectors with HNSW and efficiently cluster them to keep the most representative fake clusters
    """
    
    def __init__(
        self,
        raw_vectors: torch.Tensor,
        labels: list[int],
        fake_ratio_threshold: float = 0.8,
        similarity_threshold: float = 0.99,
        min_cluster_size: int = 5,
        top_k: int = 100,
        ef_construction: int = 300,
        ef_search: int = 300,
        m: int = 32,
        embedding_dim = 128
    ):
        self.fake_ratio_threshold = fake_ratio_threshold
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.top_k = top_k

        # L2 normalize the vectors for cosine similarity
        normalized_vectors = raw_vectors.float().view(raw_vectors.size(0), -1)
        normalized_vectors = normalized_vectors / normalized_vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)\

        self.vectors = normalized_vectors
        self.labels = list(labels)

        self.clusters = {i: {i} for i in range(self.vectors.size(0))} # initial clusters with each vector as its own cluster
        self.cluster_of = list(range(self.vectors.size(0)))
            
        self.hnsw = hnsw.Index(space="cosine", dim=embedding_dim)
        self.hnsw.init_index(max_elements=normalized_vectors.size(0), ef_construction=ef_construction, M=m)
        self.hnsw.add_items(normalized_vectors.cpu().numpy(), np.arange(normalized_vectors.size(0)))
        self.hnsw.set_ef(ef_search)

        self.reference_vectors = self.build_clusters(vectors=self.vectors, labels=self.labels)
            
    def approximate_nearest_cluster(self, query: torch.Tensor, query_index: int) -> int:
        """
        approximate nearest-cluster assignment for one vector with similarity checks
        """
        current_size = self.hnsw.get_current_count()
        if current_size < 2:
            return -1

        query_k = min(max(1, self.top_k), current_size - 1)
        max_distance = 1.0 - self.similarity_threshold

        neighbour_ids, neighbour_distances = self.hnsw.knn_query( # top-k query
            query.unsqueeze(0).cpu().numpy(),
            k=query_k,
        )  
        neighbour_ids = neighbour_ids[0]
        neighbour_distances = neighbour_distances[0]

        # check if all neighbors satisfy the cosine-distance threshold if not all top-k are in target cluster
        # case 1: some number of the target cluster is in the top-k neighbours and potentially outside, but not provably dissimilar enough if outside
        # worst case this becomes transitive closure of all pairs above similarity threshold
        all_within_threshold = all(dist <= max_distance for dist in neighbour_distances)

        target_cids = {self.cluster_of[j] for j in neighbour_ids if self.cluster_of[j] != self.cluster_of[query_index]} # exclude self

        # iterate through target_cids until we find a suitable cluster to merge with
        for target_cid in target_cids:
            target_members = self.clusters[target_cid]
            all_topk_in_target = False

            if not all_within_threshold:  # case 2: not all top-k are similar enough, so check for strict target subset
                all_topk_in_target = (  # check if all top-k neighbors are in the target cluster and satisfy similarity threshold
                    set(neighbour_ids).issubset(target_members)
                    and all(
                        dist <= max_distance
                        for neighbour, dist in zip(neighbour_ids, neighbour_distances)
                        if neighbour in target_members
                    )
                )

            if not (all_topk_in_target or all_within_threshold):
                continue
            
            return target_cid
        
        return -1 # no suitable cluster found
            

    def build_clusters(self, vectors: torch.Tensor, labels: list[int]) -> torch.Tensor:
        """
        clustering with similarity constraint using HNSW top-k neighbourhood checks
        """
        num_vectors = vectors.shape[0]

        self.clusters = {i: {i} for i in range(num_vectors)}
        self.cluster_of = list(range(num_vectors))

        for i in range(num_vectors):
            source_cid = self.cluster_of[i]
            target_cid = self.approximate_nearest_cluster(query=vectors[i], query_index=i)
            
            if target_cid == -1: # no suitable cluster found, skip merging
                continue

            # merge source cluster into target cluster
            source_members = self.clusters[source_cid]
            for member in source_members:
                self.cluster_of[member] = target_cid
            self.clusters[target_cid].update(source_members)
            
            del self.clusters[source_cid]
        
        # filter clusters by fake ratio and size
        kept_indices: list[int] = []
        for cluster in self.clusters.values():
            if len(cluster) < self.min_cluster_size:
                continue
            fake_count = sum(1 for idx in cluster if labels[idx] == 1)
            fake_ratio = fake_count / max(1, len(cluster))
            if fake_ratio >= self.fake_ratio_threshold:
                kept_indices.extend(cluster)

        if not kept_indices:
            return vectors.new_empty((0, vectors.size(-1)))

        kept_indices = sorted(set(kept_indices))
        return vectors[kept_indices]

    def update_memory(self, new_vectors: torch.Tensor) -> None:
        """
        update memory with new vectors and rebuild reference vectors, new_vectors should be fake only
        """
        normalized_new_vectors = new_vectors.float().view(new_vectors.size(0), -1)
        normalized_new_vectors = normalized_new_vectors / normalized_new_vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)

        old_count = self.vectors.size(0)
        new_count = normalized_new_vectors.size(0)
        total_count = old_count + new_count

        self.hnsw.resize_index(total_count)
        self.hnsw.add_items(normalized_new_vectors.cpu().numpy(), np.arange(old_count, total_count))

        self.vectors = torch.cat([self.vectors, normalized_new_vectors], dim=0)
        self.labels.extend([1] * new_count)

        for idx in range(old_count, total_count):
            self.clusters[idx] = {idx}
            self.cluster_of.append(idx)

        self.reference_vectors = self.build_clusters(vectors=self.vectors, labels=self.labels)
