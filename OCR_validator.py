import numpy as np
import difflib


class Score:
    def __init__(self):
        self.predictions_score: np.array = np.array([])

    def add_evaluation(self, score) -> None:
        self.predictions_score = np.append(self.predictions_score, score)

    def get_predictions_size(self) -> int:
        return len(self.predictions_score)

    def get_good_answers(self, threshold: float = 0.9) -> int:
        return np.sum(self.predictions_score > threshold)

    def get_avg_score(self) -> np.floating:
        return np.mean(self.predictions_score)

    def get_median_score(self) -> np.floating:
        return np.median(self.predictions_score)


class OCRValidator:
    def __init__(self) -> None:
        self.models_scores: dict[str: Score] = {}

    def evaluate(self, model, predicted, true_text) -> float:
        score = difflib.SequenceMatcher(None, predicted, true_text).ratio()

        if self.models_scores.get(model, None) is None:
            self.models_scores[model] = Score()

        self.models_scores[model].add_evaluation(score)

        return score

    def display_scores(self, threshold: float = 0.9) -> None:
        print(f"{'Model':<15}{'Evaluations':<15}{'Avg Score':<15}{'Median Score':<15}{'Good Answers':<15}")
        print("-" * 70)

        for model, score_obj in self.models_scores.items():
            num_evals = score_obj.get_predictions_size()
            avg_score = score_obj.get_avg_score()
            median_score = score_obj.get_median_score()
            good_answers = score_obj.get_good_answers(threshold)

            print(f"{model:<15}{num_evals:<15}{avg_score:<15.2f}{median_score:<15.2f}{good_answers:<15}")
