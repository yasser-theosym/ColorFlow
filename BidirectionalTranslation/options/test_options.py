from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        
        
        # Additional test-specific arguments
        parser.add_argument('--results_dir', type=str, default='../results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=30, help='how many test images to run')
        parser.add_argument('--n_samples', type=int, default=1, help='#samples')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio for the results')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--folder', type=str, default='intra', help='saves results here.')
        parser.add_argument('--sync', action='store_true', help='use the same latent code for different input images')

        self.isTrain = False
        return parser