# build_eca.py
# Implements Algorithm 3 to determine classes to include in eCA

def build_eca(classes, Ai_dict, Aib_dict, delta_thresh=0):
    L = []
    for i in classes:
        delta = Ai_dict[i] - Aib_dict[i]
        L.append((i, delta))

    L.sort(key=lambda x: x[1], reverse=True)
    L = [item for item in L if item[1] > delta_thresh]
    selected_classes = [i for i, _ in L]
    return selected_classes
