# classify_with_eca.py
# Implements Algorithm 4 to classify an input using the eCA mechanism

def classify(input_tensor, B_dict, R_dict, N_b, class_list, P_func):
    G = []
    for i in class_list:
        if B_dict[i](input_tensor) == i:
            G.append((i, P_func(B_dict[i](input_tensor))))

    if G:
        i_max = max(G, key=lambda x: x[1])[0]
        if R_dict[i_max](input_tensor) != 'OR':
            return R_dict[i_max](input_tensor)
    return N_b(input_tensor)
