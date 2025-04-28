import numpy as np

def calculate_kmer_frequency(sequence, k):
    kmer_count = {}
    total_kmers = 0

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in kmer_count:
            kmer_count[kmer] += 1
        else:
            kmer_count[kmer] = 1
        total_kmers += 1

    kmer_frequency = {kmer: count / total_kmers for kmer, count in kmer_count.items()}
    return kmer_frequency

def calculate_pearson_correlation(seq1, seq2, k=4):
    freq1 = calculate_kmer_frequency(seq1, k)
    freq2 = calculate_kmer_frequency(seq2, k)

    common_kmers = set(freq1.keys()) & set(freq2.keys())

    if len(common_kmers) == 0:
        return 0.0

    x = np.array([freq1[kmer] for kmer in common_kmers])
    y = np.array([freq2[kmer] for kmer in common_kmers])

    correlation = np.corrcoef(x, y)[0, 1]
    return correlation

def compare_sequence_files(gen_sequence, natural_sequence, k=4):
    correlation = calculate_pearson_correlation(gen_sequence, natural_sequence, k)
    return correlation
