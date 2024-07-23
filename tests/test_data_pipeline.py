import unittest
from src.data.data_pipeline import make_atom_features


class TestAtomFeatures(unittest.TestCase):
    def setUp(self):
        self.sequence = "SHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQD" \
                        "KAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKP" \
                        "GLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEEL" \
                        "MVDNWRPAQPLKNRQIKASFK"
        self.chain_id = "A"

    def test_make_atom_features(self):
        features = make_atom_features(self.sequence, self.chain_id)
        self.assertTrue(isinstance(features, dict))
