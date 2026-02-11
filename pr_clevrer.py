import json

from stable_datasets.images.clevrer import CLEVRER

print("Loading CLEVRER dataset...")
clevrer_train = CLEVRER(split="train")
clevrer_val = CLEVRER(split="validation")
clevrer_test = CLEVRER(split="test")
clevrer_all = CLEVRER(split=None)

print(f"\nDataset Metadata:")
print(f"  - Homepage: {clevrer_train.info.homepage}")
print(f"  - Description: {clevrer_train.info.description}")
print(f"  - Citation:\n{clevrer_train.info.citation}")

print(f"\nDataset Statistics:")
print(f"  - Train samples: {len(clevrer_train)}")
print(f"  - Validation samples: {len(clevrer_val)}")
print(f"  - Test samples: {len(clevrer_test)}")
print(f"  - Total splits: {len(clevrer_all)}")
print(f"  - Total samples: {len(clevrer_all['train']) + len(clevrer_all['validation']) + len(clevrer_all['test'])}")

sample = clevrer_train[0]
print(f"\nSample Information:")
print(f"  - Keys: {list(sample.keys())}")
print(f"  - Video type: {type(sample['video'])}")
print(f"  - Scene index: {sample['scene_index']}")
print(f"  - Video filename: {sample['video_filename']}")

# Parse questions JSON
questions = json.loads(sample["questions_json"])
print(f"\nQuestions Information:")
print(f"  - Number of questions: {len(questions)}")
if len(questions) > 0:
    q = questions[0]
    print(f"  - First question: {q['question']}")
    print(f"  - Question type: {q['question_type']}")
    print(f"  - Answer: {q.get('answer', 'N/A')}")

# Parse annotations JSON
annotations = json.loads(sample["annotations_json"])
print(f"\nAnnotations Information:")
if annotations:
    print(f"  - Number of objects: {len(annotations.get('object_property', []))}")
    print(f"  - Number of collisions: {len(annotations.get('collision', []))}")
    if annotations.get('object_property'):
        obj = annotations['object_property'][0]
        print(f"  - First object: {obj['color']} {obj['material']} {obj['shape']}")
else:
    print("  - No annotations available (test split)")

print("\nCLEVRER dataset loaded successfully!")
