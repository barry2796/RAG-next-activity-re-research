import datetime
import difflib
import os
import re
import time
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class VerificationOracle:
    def __init__(self, info_run, earlyness_boundaries=None):
        self.true_next_activities = []
        self.predicted_next_activities = []
        self.total_classes = list(info_run['Activities'])
        self.considered_classes = []
        self.prefix_with_expected_answer_pairs = {}
        self.accuracy = 0
        self.precision_macro = 0
        self.recall_macro = 0
        self.f1score_macro = 0
        self.results = []
        self.run_info = info_run
        self.start_time = time.time()
        self.end_time = 0
        self.elapsed_time = 0
        self.avg_time_per_prediction = 0
        
        self.prefix_lengths = []
        self.earlyness_buckets = defaultdict(lambda: {
            'true_activities': [],
            'predicted_activities': [],
            'results': []
        })
        self.earlyness_metrics = {}
        self.earlyness_boundaries = earlyness_boundaries or [5, 10, 20, 30]


    def add_prefix_with_expected_answer_pair(self, prefix, expected_answer):
        self.prefix_with_expected_answer_pairs[prefix] = expected_answer


    def _calculate_prefix_length(self, prefix):
        activity_part = prefix.split(' - Values:')[0]
        activities = activity_part.split(',')
        return len(activities)
    

    def _get_earlyness_bucket(self, prefix_length):
        boundaries = self.earlyness_boundaries
        
        for i, boundary in enumerate(boundaries):
            if prefix_length <= boundary:
                if i == 0:
                    return f"Very Early (1-{boundary})"
                else:
                    prev_boundary = boundaries[i-1]
                    return f"Bucket {i+1} ({prev_boundary+1}-{boundary})"
        last_boundary = boundaries[-1]
        return f"Very Late ({last_boundary+1}+)"
    

    def _get_bucket_order(self):
        boundaries = self.earlyness_boundaries
        bucket_order = []
        
        for i, boundary in enumerate(boundaries):
            if i == 0:
                bucket_order.append(f"Very Early (1-{boundary})")
            else:
                prev_boundary = boundaries[i-1]
                bucket_order.append(f"Bucket {i+1} ({prev_boundary+1}-{boundary})")

        last_boundary = boundaries[-1]
        bucket_order.append(f"Very Late ({last_boundary+1}+)")
        
        return bucket_order


    def verify_answer(self, prompt, prefix, model_answer):
        prefix_length = self._calculate_prefix_length(prefix)
        earlyness_bucket = self._get_earlyness_bucket(prefix_length)
        
        result = {
            'prompt': prompt,
            'prefix': prefix,
            'prefix_length': prefix_length,
            'earlyness_bucket': earlyness_bucket,
            'model_answer': model_answer,
            'expected_answer': None,
            'verification_result': None
        }
        expected_answer = self.prefix_with_expected_answer_pairs.get(prefix)
        if expected_answer is not None:
            result["expected_answer"] = expected_answer

            match = re.search(r'\\boxed\{([^}]*)\}', model_answer)
            if not match:
                match = re.search(r'### Assistant:\s*(.*)', model_answer)
            if not match:
                matches = list(re.finditer(r'\| Next activity:\s*(.*)', model_answer))
                if matches:
                    match = matches[-1]  # Last one
            if match:
                content = match.group(1).strip("'")

                if "None" in content:
                    model_answer = '\\boxed{Wrong format}'
                else:
                    # Remove \text{} and special characters
                    content = re.sub(r'\\text\{(.*?)\}', r'\1', content)
                    content = re.sub(r'[^a-zA-Z0-9\s]', '', content).strip()

                    # Find closest class match
                    closest_match = difflib.get_close_matches(content, self.total_classes, n=1, cutoff=0.6)
                    if closest_match:
                        model_answer = closest_match[0]
                    else:
                        model_answer = '\\boxed{Wrong format}'
            else:
                model_answer = '\\boxed{Wrong format}'

            result["verification_result"] = (
                    expected_answer.lower().replace(" ", "") in model_answer.lower().replace(" ", "")
            )
            print(
                f"Prompt: {prompt}\n"
                f"Answer: {model_answer}\n"
                f"Expected Answer: {expected_answer}\n"
                f"Prefix Length: {prefix_length}\n"
                f"Earlyness Bucket: {earlyness_bucket}\n"
                f"Result: {result['verification_result']}"
            )
            if result["verification_result"]:
                model_answer = expected_answer

        self.results.append(result)
        self.true_next_activities.append(expected_answer.removeprefix("\\boxed{").removesuffix("}"))
        self.predicted_next_activities.append(model_answer.removeprefix("\\boxed{").removesuffix("}"))
        self.prefix_lengths.append(prefix_length)
        
        self.earlyness_buckets[earlyness_bucket]['true_activities'].append(
            expected_answer.removeprefix("\\boxed{").removesuffix("}"))
        self.earlyness_buckets[earlyness_bucket]['predicted_activities'].append(
            model_answer.removeprefix("\\boxed{").removesuffix("}"))
        self.earlyness_buckets[earlyness_bucket]['results'].append(result)

        return result["verification_result"]


    def compute_stats(self):
        self.considered_classes = list(set(self.true_next_activities + self.predicted_next_activities))
        self.precision_macro = precision_score(self.true_next_activities, self.predicted_next_activities,
                                               labels=self.considered_classes, average='macro')
        self.recall_macro = recall_score(self.true_next_activities, self.predicted_next_activities, labels=self.considered_classes,
                                         average='macro')
        self.f1score_macro = f1_score(self.true_next_activities, self.predicted_next_activities, labels=self.considered_classes,
                                      average='macro')
        self.accuracy = accuracy_score(self.true_next_activities,
                                       self.predicted_next_activities)
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.avg_time_per_prediction = self.elapsed_time / len(self.true_next_activities) if len(self.true_next_activities) > 0 else 0
        self._compute_earlyness_metrics()
    

    def _compute_earlyness_metrics(self):
        for bucket, data in self.earlyness_buckets.items():
            if len(data['true_activities']) == 0:
                continue
                
            true_acts = data['true_activities']
            pred_acts = data['predicted_activities']
            considered_classes_bucket = list(set(true_acts + pred_acts))
            
            if len(considered_classes_bucket) > 1:
                precision = precision_score(true_acts, pred_acts, 
                                          labels=considered_classes_bucket, 
                                          average='macro', zero_division=0)
                recall = recall_score(true_acts, pred_acts, 
                                    labels=considered_classes_bucket, 
                                    average='macro', zero_division=0)
                f1 = f1_score(true_acts, pred_acts, 
                            labels=considered_classes_bucket, 
                            average='macro', zero_division=0)
            else:
                precision = recall = f1 = 1.0 if true_acts == pred_acts else 0.0    # If only one class
                
            accuracy = accuracy_score(true_acts, pred_acts)
            
            self.earlyness_metrics[bucket] = {
                'count': len(true_acts),
                'accuracy': accuracy,
                'precision_macro': precision,
                'recall_macro': recall,
                'f1score_macro': f1,
                'considered_classes': considered_classes_bucket
            }


    def write_results_to_file(self):
        file_path = os.path.join(os.path.dirname(__file__), "..", "tests", "validation",
                                 f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
        self.compute_stats()

        with open(file_path, 'w') as file:
            file.write('INFORMATION ON THE RUN\n\n')
            for key in self.run_info.keys():
                file.write(f"{key}: {self.run_info[key]}\n")
            file.write('\n-----------------------------------\n')
            file.write('OVERALL METRICS\n')
            file.write(f"Accuracy: {self.accuracy:.4f}\n")
            file.write(f"Precision (macro): {self.precision_macro:.4f}\n")
            file.write(f"Recall (macro): {self.recall_macro:.4f}\n")
            file.write(f"F1-score (macro): {self.f1score_macro:.4f}\n")
            file.write(f"Total classes: {self.total_classes}\n")
            file.write(f"Elapsed time: {(self.elapsed_time / 3600):.2f} hours\n")
            file.write(f"Average time per prediction: {self.avg_time_per_prediction:.4f} seconds\n")
            file.write('\n-----------------------------------\n')
            file.write('EARLYNESS ANALYSIS\n')
            file.write('Metrics by Prefix Length Buckets:\n\n')

            bucket_order = self._get_bucket_order()
            for bucket in bucket_order:
                if bucket in self.earlyness_metrics:
                    metrics = self.earlyness_metrics[bucket]
                    file.write(f"{bucket}:\n")
                    file.write(f"  Sample Count: {metrics['count']}\n")
                    file.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                    file.write(f"  Precision (macro): {metrics['precision_macro']:.4f}\n")
                    file.write(f"  Recall (macro): {metrics['recall_macro']:.4f}\n")
                    file.write(f"  F1-score (macro): {metrics['f1score_macro']:.4f}\n")
                    file.write(f"  Classes in bucket: {len(metrics['considered_classes'])}\n\n")
            
            if self.prefix_lengths:
                file.write(f"Prefix Length Statistics:\n")
                file.write(f"  Min length: {min(self.prefix_lengths)}\n")
                file.write(f"  Max length: {max(self.prefix_lengths)}\n")
                file.write(f"  Average length: {sum(self.prefix_lengths)/len(self.prefix_lengths):.2f}\n\n")
            
            file.write("-----------------------------------\n\n")

            for result in self.results:
                file.write(f"Prompt: {result['prompt']}\n")
                file.write(f"Prefix Length: {result['prefix_length']}\n")
                file.write(f"Earlyness Bucket: {result['earlyness_bucket']}\n")
                file.write(f"Model Answer: {result['model_answer']}\n")
                file.write(f"Expected Answer: {result['expected_answer']}\n")
                file.write(f"Verification Result: {result['verification_result']}\n")
                file.write("\n#####################################################################################\n")

    def get_earlyness_summary(self):
        if not self.earlyness_metrics:
            self._compute_earlyness_metrics()
        
        summary = {
            'overall_metrics': {
                'accuracy': self.accuracy,
                'precision_macro': self.precision_macro,
                'recall_macro': self.recall_macro,
                'f1score_macro': self.f1score_macro,
                'total_samples': len(self.true_next_activities),
                'avg_time_per_prediction': getattr(self, 'avg_time_per_prediction', 0)
            },
            'earlyness_metrics': self.earlyness_metrics,
            'prefix_length_stats': {
                'min': min(self.prefix_lengths) if self.prefix_lengths else 0,
                'max': max(self.prefix_lengths) if self.prefix_lengths else 0,
                'average': sum(self.prefix_lengths)/len(self.prefix_lengths) if self.prefix_lengths else 0
            }
        }
        return summary
