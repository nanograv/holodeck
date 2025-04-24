from holodeck.discrete import population, evolution
import numpy as np

class Triples(population.Pop_Illustris):
    """
    Class to find the occurance of triples in the cosmological simulation binary population.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Triples class.
        """
        super().__init__(*args, **kwargs)
        print(self._fname)

        #call the blackhole ids
        self.bhids = self.bh_merger_file['blackhole-mergers/bhid'][self.valid_merger_flag]
        #the redshift of binary formations
        self.z_binary_form = 1/self.bh_merger_file['blackhole-mergers/scafa'][self.valid_merger_flag] - 1

        unique_ids, counts = np.unique(self.bhids, return_counts=True)
        self.repeated_ids = unique_ids[counts > 1]
        print(f"% of binaries with multiple mergers: {(len(self.repeated_ids)/self.N_binaries)*100:.2f}%")

        









