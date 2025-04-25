from holodeck.discrete import population, evolution
import numpy as np

class Triples:
    """
    -----------
    This class is used to find the triples in the cosmological simulation binary population. Primarily to be used with :class:`holodeck.population.Pop_Illustris`.
    -----------
    """

    def __init__(self,pop,evo,*args, **kwargs):
        """
        Initialize the Triples class.
        """
        #super().__init__(*args, **kwargs)
        #DEBUG: print(self._fname)

        #call the blackhole ids
        self.bhids = pop.bh_merger_file['blackhole-mergers/bhid'][pop.valid_merger_flag]
        #the redshift of binary formations
        self.z_binary_form = 1/pop.bh_merger_file['blackhole-mergers/scafa'][pop.valid_merger_flag] - 1

        unique_ids, counts = np.unique(self.bhids, return_counts=True)
        self.repeated_ids = unique_ids[counts > 1]
        # print(f"% of binaries with multiple mergers: {(len(self.repeated_ids)/pop.N_binaries)*100:.2f}%")
        overlap_counter=0
        trip_idx_data = [] #to store the indices of the triple candidates

        for i in range(len(self.repeated_ids)):
            repeated_id_idxs = np.argwhere(self.bhids == self.repeated_ids[i])[:,0]
            for i in range(len(repeated_id_idxs)-1):
                first_binary_idx = repeated_id_idxs[i]
                second_binary_idx = repeated_id_idxs[i+1]

                first_binary_zform = self.z_binary_form[first_binary_idx]
                second_binary_zform = self.z_binary_form[second_binary_idx]

                first_binary_tlook_form = evo.tlook[first_binary_idx][0]
                first_binary_tlook_merger = evo.tlook[first_binary_idx][-1]

                second_binary_tlook_form = evo.tlook[second_binary_idx][0]
                second_binary_tlook_merger = evo.tlook[second_binary_idx][-1]

                if(first_binary_tlook_merger<second_binary_tlook_form):
                    overlap_counter = overlap_counter + 1
                    trip_idx_data.append([first_binary_idx,second_binary_idx])
                
        print('fraction of triple interactions',overlap_counter/pop.N_binaries)



        









