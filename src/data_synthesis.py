import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
import numpy as np
from collections import Counter
from pdb import set_trace
import faiss
import torch


class ConceptSampler:
    def __init__(self, concept_lists, concept_embeddings):
        """
        concept_lists: List[List[str]] - lists of concept names
        concept_embeddings: dict - mapping from concept name to vector
        """
        self.concept_lists = concept_lists
        self.embeddings = concept_embeddings
        # Create co-occurrence statistics
        self.build_cooccurrence_stats()

    def build_cooccurrence_stats(self):
        """Build co-occurrence matrix and concept frequencies"""
        # Get unique concepts
        self.all_concepts = list(set(c for lst in self.concept_lists for c in lst))
        self.concept_to_idx = {c: i for i, c in enumerate(self.all_concepts)}

        # Count individual frequencies
        self.concept_freq = {c: 0 for c in self.all_concepts}
        for lst in self.concept_lists:
            for c in lst:
                self.concept_freq[c] += 1

        # Build co-occurrence matrix
        n = len(self.all_concepts)
        self.cooccur_matrix = np.zeros((n, n))
        for lst in self.concept_lists:
            for c1 in lst:
                for c2 in lst:
                    if c1 != c2:
                        i, j = self.concept_to_idx[c1], self.concept_to_idx[c2]
                        self.cooccur_matrix[i, j] += 1

    def sample_concept_list(self, size=3, temperature=0.1, seed_concept=None):
        """
        Sample a list of concepts
        size: desired list size
        temperature: controls randomness (lower = more deterministic)
        seed_concept: optional starting concept
        """
        result = []

        # Start with seed concept or random concept
        if seed_concept is None:
            # Sample first concept based on frequency
            freqs = np.array([self.concept_freq[c] for c in self.all_concepts])
            probs = freqs / freqs.sum()
            seed_concept = np.random.choice(self.all_concepts, p=probs)

        result.append(seed_concept)

        # Sample remaining concepts
        while len(result) < size:
            scores = self.get_next_concept_scores(result)

            # Apply temperature
            scores = np.exp(scores / temperature)
            probs = scores / scores.sum()

            # Sample next concept
            next_concept = np.random.choice(self.all_concepts, p=probs)
            result.append(next_concept)

        return result

    def get_next_concept_scores(self, current_list):
        """Calculate scores for potential next concepts"""
        scores = np.zeros(len(self.all_concepts))

        for concept in self.all_concepts:
            if concept in current_list:
                continue

            # Co-occurrence score
            cooccur_score = 0
            for c in current_list:
                i, j = self.concept_to_idx[concept], self.concept_to_idx[c]
                cooccur_score += self.cooccur_matrix[i, j]

            # Semantic similarity score
            sim_score = 0
            concept_vec = self.embeddings[concept]
            for c in current_list:
                sim = cosine_similarity(
                    [concept_vec],
                    [self.embeddings[c]]
                )[0][0]
                sim_score += sim

            # Combine scores
            scores[self.concept_to_idx[concept]] = cooccur_score + sim_score

        return scores


class FastConceptSampler:
    def __init__(self, concept_lists, concept_embeddings):
        self.concept_lists = concept_lists
        self.embeddings = concept_embeddings
        self.embedding_keys = set(self.embeddings.keys())
        self.build_cooccurrence_stats()
        self.setup_faiss_index()

    def build_cooccurrence_stats(self):
        # Get unique concepts and create mapping
        self.all_concepts = list(set(c for lst in self.concept_lists for c in lst))
        print(len(self.all_concepts))
        self.all_concepts = list(filter(lambda x: x in self.embedding_keys, self.all_concepts))
        # from pdb import set_trace; set_trace()
        # arr_1d = list(set([item for sublist in self.concept_lists for item in sublist]))
        self.concept_to_idx = {c: i for i, c in enumerate(self.all_concepts) if c in self.embedding_keys}
        self.idx_to_concept = {i: c for c, i in self.concept_to_idx.items()}

        # Count frequencies using Counter
        self.concept_freq = Counter(c for lst in self.concept_lists for c in lst)

        # Build sparse co-occurrence matrix
        rows, cols, data = [], [], []
        for lst in self.concept_lists:
            lst = list(filter(lambda x: x in self.embedding_keys, lst))
            lst_idx = [self.concept_to_idx[c] for c in lst]
            for i in lst_idx:
                for j in lst_idx:
                    if i != j:
                        rows.append(i)
                        cols.append(j)
                        data.append(1)

        n = len(self.all_concepts)
        self.cooccur_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))

    def setup_faiss_index(self):
        # Convert embeddings to numpy array
        n = len(self.all_concepts)
        d = len(next(iter(self.embeddings.values())))
        embedding_matrix = np.zeros((n, d), dtype=np.float32)

        for concept, idx in self.concept_to_idx.items():
            embedding_matrix[idx] = self.embeddings[concept]
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix /= norms

        # Create FAISS index
        self.index = faiss.IndexFlatIP(d)  # Inner product = cosine similarity for normalized vectors
        self.index.add(embedding_matrix)

    def sample_concept_list(self, size=3, temperature=0.1, seed_concept=None):
        result = []

        if seed_concept is None:
            # Sample based on frequency using numpy
            concepts = np.array(self.all_concepts)
            freqs = np.array([self.concept_freq[c] for c in concepts])
            probs = freqs / freqs.sum()
            seed_concept = np.random.choice(concepts, p=probs)

        result.append(seed_concept)

        # Pre-compute indices for current list
        used_indices = {self.concept_to_idx[c] for c in result}

        while len(result) < size:
            scores = self.get_next_concept_scores_fast(result, used_indices)

            # Temperature scaling and sampling
            scores = np.exp(scores / temperature)
            probs = scores / scores.sum()

            next_idx = np.random.choice(len(self.all_concepts), p=probs)
            next_concept = self.idx_to_concept[next_idx]

            result.append(next_concept)
            used_indices.add(next_idx)

        return result

    def get_next_concept_scores_fast(self, current_list, used_indices):
        n = len(self.all_concepts)
        scores = np.zeros(n, dtype=np.float32)

        # Compute co-occurrence scores using sparse matrix
        current_indices = [self.concept_to_idx[c] for c in current_list]
        cooccur_scores = self.cooccur_matrix[current_indices].sum(axis=0).A1

        # Compute similarity scores using FAISS
        query = np.mean([self.embeddings[c] for c in current_list], axis=0)
        query = query.reshape(1, -1).astype(np.float32)
        query /= np.linalg.norm(query)
        sim_scores, _ = self.index.search(query, n)
        sim_scores = sim_scores[0]

        # Combine scores
        scores = cooccur_scores + sim_scores

        # Mask out used concepts
        scores[list(used_indices)] = -np.inf

        return scores


def thinking_process_generation_prompt(problem, concepts, difficulty_level):
    concept_text = "\n".join(f"{i+1}. {concept}" for i, concept in enumerate(concepts))
    prompt = (
        "Imagine you are an expert in educational problem design.\n"
        f"You will be shown these components:\n\n"
        f"Problem: {problem}\n\n"
        f"Foundamental Concepts:\n{concept_text}\n\n"
        f"Difficulty Level: {difficulty_level}\n\n"
        "Your task is to reverse-engineer a clear thinking process that shows how a teacher might design this problem. This thinking process should:\n"
        "- Show how combining the given foundational concepts naturally leads to a problem at the specified difficulty level\n"
        "- Include all key decisions and reasoning that shaped the problem design\n"
        "- (IMPORTANT) The thinking process must be so precise and detailed that another teacher following these exact steps would recreate the identical problem\n"
        "- (IMPORTANT) The thinking process must be so natural and logical that another teacher could derive the same thinking process using only the foundational concepts and difficulty level\n\n"
        "Present your answer after 'Thinking Process: ' with the complete step-by-step thinking process described above."
    )

    return prompt


def topic_extraction_prompt(problem, num_concepts):
    prompt = (
        "As an expert in educational assessment, analyze this problem:\n\n"
        f"{problem}\n\n"
        f"Break down and identify {num_concepts} foundational concepts being tested. List these knowledge points that:\n"
        "- Are core curriculum concepts typically taught in standard courses\n"
        "- Are precise and measurable (not vague like 'understanding math')\n"
        "- Are essential building blocks needed to solve this problem\n"
        "- Represent fundamental principles rather than problem-specific techniques\n\n"
        f"Think through your analysis step by step, then format your response as a Python code snippet containing a list of {num_concepts} strings, where each string clearly describes one fundamental knowledge point."
    )
    return prompt


def get_catch_all_prompt(problem, num_concepts):
    prompt = (
        "As an expert in educational assessment, analyze this problem:\n\n"
        f"{problem}\n\n"
        f"Break down and identify {num_concepts} foundational concepts being tested. List these knowledge points that:\n"
        "- Are core curriculum concepts typically taught in standard courses\n"
        "- Are precise and measurable (not vague like 'understanding math')\n"
        "- Are essential building blocks needed to solve this problem\n"
        "- Represent fundamental principles rather than problem-specific techniques\n\n"
        f"Return only {num_concepts} lines, each starting with 'Knowledge Point: ' followed by one fundamental concept, without any other text."
    )
    return prompt


def rationale_judgement_prompt(concepts, level, rationale_and_problem):
    concept_text = "\n".join(f"- {concept}" for i, concept in enumerate(concepts))
    prompt = (
        "As a critical expert in educational problem design, evaluate the following problem components:\n\n"
        f"=== GIVEN MATERIALS ===\n"
        f"1. Problem & Design Rationale:\n{rationale_and_problem}\n"
        "   (The rationale describes the author's thinking process and justification in designing this problem)\n\n"
        f"2. Foundational Concepts:\n{concept_text}\n\n"
        f"3. Target Difficulty Level: {level}\n\n"

        "=== EVALUATION CRITERIA ===\n"
        "Rate each criterion as: [Perfect | Acceptable | Bad]\n\n"

        "1. FORMAT\n"
        "- Verify correct implementation of markup tags:\n"
        "  <!-- BEGIN RATIONALE --> [design thinking process] <!-- END RATIONALE -->\n"
        "  <!-- BEGIN PROBLEM --> [problem] <!-- END PROBLEM -->\n\n"

        "2. FACTUAL ACCURACY\n"
        "- Check for any incorrect or misleading information in both problem and rationale\n"
        "- Verify mathematical, scientific, or logical consistency\n\n"

        "3. DIFFICULTY ALIGNMENT\n"
        "- Assess if problem complexity matches the specified difficulty level\n"
        "- Evaluate if cognitive demands align with target level\n\n"

        "4. CONCEPT COVERAGE\n"
        "- Evaluate how well the problem incorporates the given foundational concepts\n"
        "- Check for missing concept applications\n\n"

        "5. SOLVABILITY\n"
        "- Verify if the problem has at least one valid solution\n"
        "- Check if all necessary information for solving is provided\n\n"

        "=== RESPONSE FORMAT ===\n"
        "For each criterion, provide:\n"
        "1. Rating: [Perfect | Acceptable | Bad]\n"
        "2. Justification: Clear explanation for the rating\n\n"

        "=== FINAL VERDICT ===\n"
        "After providing all criterion evaluations, conclude your response with:\n"
        "'Final Judgement: [verdict]'\n"
        "where verdict must be one of:\n"
        "- 'perfect' (if both FACTUAL ACCURACY and SOLVABILITY are Perfect, at least two other criteria are Perfect, and no Bad ratings)\n"
        "- 'acceptable' (if no Bad ratings and doesn't qualify for perfect)\n"
        "- 'bad' (if ANY Bad ratings)\n\n"
        "Note: The 'Final Judgement: [verdict]' line must be the final line of your response."
    )

    return prompt


def extract_knowledge_points(response, keyword="Knowledge Point:"):
    knowledge_points = []
    for line in response.split("\n"):
        if len(line.split(keyword)) >= 2:
            knowledge_points.append(line.split(keyword)[-1].strip())
    return knowledge_points


def extract_thinking_process(response, keyword="Thinking Process:"):
    if len(response.split(keyword)) >= 2:
        return response.split(keyword)[-1].strip()
    else:
        return response


def count_ratings(response):
    ratings = ["perfect", "acceptable", "bad"]
    counts = {rating: response.lower().count(rating) for rating in ratings}
    max_count = max(counts.values())
    max_ratings = [rating for rating, count in counts.items() if count == max_count]
    return max_ratings


def extract_final_judgement(response, keyword="Final Judgement:"):
    if len(response.split(keyword)) >= 2:
        init_response = response.split(keyword)[-1].strip().lower()
        for res in ["perfect", "acceptable", "bad"]:
            if res in init_response:
                return res
        return init_response
    else:
        ratings = count_ratings(response)
        return ratings[0]

