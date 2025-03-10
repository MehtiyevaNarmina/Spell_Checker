import nltk, time
from nltk.corpus import words
from concurrent.futures import ThreadPoolExecutor

nltk.download("words") # caching technique (optiization)

DICTIONARY = set(words.words())

def levenshtein_distance_dp(a, b):

    m, n = len(a), len(b)
    distance_matrix = [[0] * (n + 1) for k in range(m + 1)]    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                distance_matrix[i][j] = j
            elif j == 0:
                distance_matrix[i][j] = i
            else:
                if a[i-1] == b[j-1]:
                    cost = 0
                else:
                    cost = 1
                distance_matrix[i][j] = min(distance_matrix[i-1][j] + 1,   
                                distance_matrix[i][j-1] + 1,      
                                distance_matrix[i-1][j-1] + cost) 
    
    return distance_matrix[m][n]


def damerau_levenshtein_distance(a, b):

    len1, len2 = len(a), len(b)
    distance_matrix = [[0] * (len2 + 1) for k in range(len1 + 1)]

    for i in range(len1 + 1):
        distance_matrix[i][0] = i
    for j in range(len2 + 1):
        distance_matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            distance_matrix[i][j] = min(
                distance_matrix[i-1][j] + 1,        
                distance_matrix[i][j-1] + 1,        
                distance_matrix[i-1][j-1] + cost    
            )

            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                distance_matrix[i][j] = min(distance_matrix[i][j], distance_matrix[i-2][j-2] + cost)

    return distance_matrix[len1][len2]


def jaro_similarity(a, b):

    len1, len2 = len(a), len(b)
    
    if len1 == 0 and len2 == 0:
        return 1.0
    
    match_distance = (max(len1, len2) // 2) - 1
    
    matches1 = [0] * len1
    matches2 = [0] * len2
    
    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(len2, i + match_distance + 1)
        for j in range(start, end):
            if not matches2[j] and a[i] == b[j]:
                matches1[i] = matches2[j] = True
                matches += 1
                break
    
    if matches == 0:
        return 0.0
    
    transpositions = 0
    k = 0
    for i in range(len1):
        if matches1[i]:
            while not matches2[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    
    jaro_sim = (
        (matches / len1) +
        (matches / len2) +
        ((matches - transpositions // 2) / matches)
    ) / 3
    
    return jaro_sim


def jaro_winkler_similarity(a, b):

    precomputed_jaro = jaro_similarity(a, b)  

    prefix_length = 0
    max_prefix = 4
    for i in range(min(len(a), len(b), max_prefix)):
        if a[i] == b[i]:
            prefix_length += 1
        else:
            break

    jaro_winkler = precomputed_jaro + (prefix_length * 0.1 * (1 - precomputed_jaro))

    return jaro_winkler

# function map (optimization)
distance_functions = {
    "levenshtein": levenshtein_distance_dp,
    "damerau_levenshtein": damerau_levenshtein_distance,
}
similarity_functions = {
    "jaro_similarity": jaro_similarity,
    "jaro_winkler": jaro_winkler_similarity,
}

def compute_distance(word, correct_word, method):
    if method in distance_functions:
        return correct_word, distance_functions[method](word, correct_word)
    similarity = similarity_functions[method](word, correct_word)
    return correct_word, 1 - similarity

def spell_check(word, dictionary, method, max_suggestions=3, max_distance=2):

    word_len = len(word)
    precomputed_lengths = {w: len(w) for w in dictionary} # precompuation (optimization)
    
    # parallel processing (optimization)
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(compute_distance, word, correct_word, method)
            for correct_word in dictionary
            if abs(word_len - precomputed_lengths[correct_word]) <= max_distance #early filtering (optimization)
        ]
        suggestions = [
            (correct_word, distance)
            for future in futures
            for correct_word, distance in [future.result()]
            if distance <= max_distance
        ]
    return [w for w, k in sorted(suggestions, key=lambda x: x[1])[:max_suggestions]]
