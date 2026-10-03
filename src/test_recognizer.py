import numpy as np

from recognizer import (
    register_face,
    find_best_match,
    recognize_or_register
)


# Create a fake 512-dimensional embedding
embedding = np.random.rand(512).astype(np.float32)

print("\n--- TEST 1: Register face ---")

face_id = register_face(embedding)

print(f"Registered Face ID: {face_id}")


print("\n--- TEST 2: Find same face ---")

best_face_id, similarity = find_best_match(embedding)

print(f"Best Face ID: {best_face_id}")
print(f"Similarity: {similarity:.4f}")


print("\n--- TEST 3: Recognize same face ---")

recognized_id = recognize_or_register(embedding)

print(f"Recognized Face ID: {recognized_id}")


print("\n✅ Recognizer test completed")