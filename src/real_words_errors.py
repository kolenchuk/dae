from typing import List
import json
import os


class RealWorldErrors:
    def __init__(self, log_file: str = None):
        self.error_patterns = {
            # Default patterns from launch
            'samsung': ['samung', 'samsng', 'sumsang', 'samsum', 'sumsung'],
            'xiaomi': ['xiomi', 'xaomi', 'xiami', 'xiami'],
            'iphone': ['ipone', 'iphne', 'ifone', 'iphon'],
            'huawei': ['huawey', 'huwei', 'huawai', 'huawey'],
            'realme': ['realmi', 'relme', 'realm', 'realme'],
            'nothing': ['nothin', 'noting', 'noting'],
            'oneplus': ['onplus', '1plus', 'oneplus'],
        }

        if log_file and os.path.exists(log_file):
            self.load_from_logs(log_file)

    def load_from_logs(self, log_file: str):
        """Load real-world error patterns from logs"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        correct = log_entry['correct'].lower()
                        error = log_entry['error'].lower()

                        if correct not in self.error_patterns:
                            self.error_patterns[correct] = []
                        if error not in self.error_patterns[correct]:
                            self.error_patterns[correct].append(error)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading log file: {e}")

    def save_patterns(self, file_path: str):
        """Save current error patterns to file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.error_patterns, f, indent=2, ensure_ascii=False)

    def get_common_errors(self, word: str) -> List[str]:
        """Get known error patterns for a word"""
        return self.error_patterns.get(word.lower(), [])