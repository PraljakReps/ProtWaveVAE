

def compute_hamming_dist(
        seq1_list: list,
        seq2_list: list,
    ) -> (
            list,
            list
    ):

        L = len(seq1_list[0]) # assume length of seq1 and seq2 matches

        hamming_distance_list = []
        similarity = []
        for (seq1, seq2) in zip(seq1_list, seq2_list):

            hamming_distance =0
            for ii, (aa1, aa2) in enumerate(zip(seq1, seq2)):

                if aa1 != aa2:
                    hamming_distance+=1

                else:
                    pass
            
            hamming_distance_list.append(hamming_distance)
            similarity.append( hamming_distance / L )
        

        return (
                hamming_distance_list,
                similarity
        )


