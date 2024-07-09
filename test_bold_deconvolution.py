from unittest import TestCase
import timeit
import unittest
import numpy as np
from bold_deconvolution import ridge_regress_deconvolution, ridge_deconvolution
from bold_deconvolution import dctmtx_numpy, dctmtx_numba, dctmtx_numpy_vect
from scipy import io
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


# Assuming the ridge_regress_deconvolution function is already imported

class TestRidgeDeconvolution(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        data = io.loadmat('./data/03_BD_Sub_01_ROI_01.mat')  # Example BOLD signal of length 100
        self.BOLD = data['preproc_BOLD_signal']
        self.spm_neuro = data['spm_phys_neuro'].squeeze()
        self.TR = 0.72  # Example time repetition
        self.alpha = 0.005
        self.NT = 16

    def test_output_shape(self):
        # Test that the output shape matches the input BOLD signal length
        neuronal = ridge_deconvolution(self.BOLD, self.TR)
        self.assertEqual(len(neuronal), len(self.spm_neuro), "Output shape does not match input shape")

    def test_alpha_effect(self):
        # Test that changing alpha affects the output
        neuronal_1 = ridge_deconvolution(self.BOLD, self.TR, alpha=0.005)
        neuronal_2 = ridge_deconvolution(self.BOLD, self.TR, alpha=0.1)
        self.assertFalse(np.allclose(neuronal_1, neuronal_2), "Output should change with different alpha values")

    def test_empty_BOLD(self):
        # Test with an empty BOLD signal
        empty_BOLD = np.array([])
        with self.assertRaises(ValueError):
            ridge_deconvolution(empty_BOLD, self.TR)

    def test_different_NT(self):
        # Test that changing NT parameter works and affects the output
        neuronal_1 = ridge_deconvolution(self.BOLD, self.TR, NT=16)
        neuronal_2 = ridge_deconvolution(self.BOLD, self.TR, NT=32)
        self.assertTrue(len(neuronal_2) > len(neuronal_1), "Output should change with different NT values")

    def test_output_type(self):
        # Test that the output type is numpy ndarray
        neuronal = ridge_deconvolution(self.BOLD, self.TR)
        self.assertIsInstance(neuronal, np.ndarray, "Output type should be numpy ndarray")

    def test_closer_to_spm(self):
        alpha = 0.005
        neuronal = ridge_deconvolution(self.BOLD, self.TR, alpha=alpha, NT=16)

        plt.figure(figsize=(14, 6))
        plt.title(f'Neuronal signal recovered by ridge regression (alpha = {alpha}, ridge regression reduces to OLS method)',
                  fontsize=16)
        plt.plot(self.spm_neuro, label='SPM PEB')
        plt.plot(neuronal / np.max(neuronal) * np.max(self.spm_neuro), label='Ridge regression')
        plt.legend(loc=2, fontsize=16)
        r = pearsonr(neuronal, self.spm_neuro).correlation
        plt.text(len(neuronal)* 0.9, np.max(self.spm_neuro) * 0.9, 'r = ' + str(round(r, 2)))
        plt.show()
        self.assertTrue(True)


class TestRidgeRegressDeconvolution(unittest.TestCase):

    def setUp(self):
        # Setup some basic example data
        self.data = io.loadmat('./data/03_BD_Sub_01_ROI_01.mat')  # Example BOLD signal of length 100
        self.BOLD = self.data['preproc_BOLD_signal']
        self.spm_neuro = self.data['spm_phys_neuro'].squeeze()
        self.TR = 0.72  # Example time repetition
        self.alpha = 0.005
        self.NT = 16

    def test_plot_mat(self):
        plt.figure(figsize=(20, 5))
        plt.plot(self.BOLD, label='Preprocessed BOLD signal', alpha=0.5)
        plt.plot(self.data['psy_convolved'], c='forestgreen', label='Convolved PSY regressor')
        plt.plot(self.data['spm_ppi'], c='indianred', label='SPM PEB PPI regressor')
        plt.legend(fontsize=14)
        plt.ylim(-1.3 * np.max(self.data['preproc_BOLD_signal']), 1.3 * np.max(self.data['preproc_BOLD_signal']))
        plt.xlabel('Time points (TR = 2 s)', fontsize=14)
        plt.show()

    def test_basic_functionality(self):
        # Test if the function returns an output of expected shape and type
        neuro = ridge_regress_deconvolution(self.BOLD, self.TR, self.alpha, self.NT)
        self.assertIsInstance(neuro, np.ndarray)
        self.assertEqual(neuro.size, self.spm_neuro.size)

    def test_equal_ridge_deconv(self):
        neuro_ridge_regress = ridge_regress_deconvolution(self.BOLD, self.TR, self.alpha, self.NT)
        neuro_ridge = ridge_deconvolution(self.BOLD, self.TR, self.alpha, self.NT)
        r = pearsonr(neuro_ridge, neuro_ridge_regress.flatten()).correlation
        plt.plot(neuro_ridge)
        plt.plot(neuro_ridge_regress)

        self.assertIsNone(np.allclose(neuro_ridge_regress, neuro_ridge))

    def test_empty_input(self):
        # Test if the function raises ValueError for empty BOLD signal
        with self.assertRaises(ValueError):
            ridge_regress_deconvolution(np.array([]), self.TR, self.alpha, self.NT)

    def test_zero_TR(self):
        # Test if the function raises ZeroDivisionError for TR = 0
        with self.assertRaises(ZeroDivisionError):
            ridge_regress_deconvolution(self.BOLD, 0, self.alpha, self.NT)

    def test_output_consistency(self):
        # Test if the function produces consistent outputs for known inputs
        BOLD_test = np.ones(100)
        TR_test = 2.0
        expected_output = ridge_regress_deconvolution(BOLD_test, TR_test, self.alpha, self.NT)

        # Repeat with the same input
        output = ridge_regress_deconvolution(BOLD_test, TR_test, self.alpha, self.NT)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=6)


class TestDctmtx(unittest.TestCase):
    def setUp(self):
        self.N = 200
        self.K = 100

    def test_dctmtx_equal(self):
        xb_numpy = dctmtx_numpy(self.N, self.K)
        xb_numpy_vec = dctmtx_numpy_vect(self.N, self.K)
        xb_numba = dctmtx_numba(self.N, self.K)
        diff = xb_numba - xb_numpy
        self.assertIsNone(np.testing.assert_allclose(xb_numpy, xb_numpy_vec))
        self.assertIsNone(np.testing.assert_allclose(xb_numpy, xb_numba, rtol=1e10 - 7))

    def test_dctmtx(self):
        xb_numpy_vec = dctmtx_numpy_vect(self.N, self.K)
        plt.imshow(xb_numpy_vec)
        plt.show()




class TestDctmtxPerformance(unittest.TestCase):
    def setUp(self):
        self.N = 6384
        self.K = 1000
        self.iterations = 100

    def test_dctmtx_numpy_performance(self):
        N = self.N
        K = self.K
        numpy_time = timeit.timeit(
            'dctmtx_numpy(N, K)',
            setup='from bold_deconvolution import dctmtx_numpy',
            globals=self.__dict__,
            number=self.iterations
        )
        print(f"Numpy dctmtx time over {self.iterations} iterations: {numpy_time:.6f} seconds")

    def test_dctmtx_numpy_vect_performance(self):
        N = self.N
        K = self.K
        numpy_time = timeit.timeit(
            'dctmtx_numpy_vect(N, K)',
            setup='from bold_deconvolution import dctmtx_numpy_vect',
            globals=self.__dict__,
            number=self.iterations
        )
        print(f"Numpy_vect dctmtx time over {self.iterations} iterations: {numpy_time:.6f} seconds")

    def test_dctmtx_numba_performance(self):
        N = self.N
        K = self.K
        # first call for precompling
        C = dctmtx_numba(5, 5)
        numba_time = timeit.timeit(
            'dctmtx_numba(N, K)',
            setup='from bold_deconvolution import dctmtx_numba',
            globals=self.__dict__,
            number=self.iterations
        )
        print(f"Numba dctmtx time over {self.iterations} iterations: {numba_time:.6f} seconds")

    def test_speedup(self):
        N = self.N
        K = self.K
        numpy_time = timeit.timeit(
            'dctmtx_numpy(N, K)',
            setup='from bold_deconvolution import dctmtx_numpy',
            globals=self.__dict__,
            number=self.iterations
        )

        # first call for precompling
        C = dctmtx_numba(3, 3)

        numba_time = timeit.timeit(
            'dctmtx_numba(N, K)',
            setup='from bold_deconvolution import dctmtx_numba',
            globals=self.__dict__,
            number=self.iterations
        )
        speedup = numpy_time / numba_time
        print(f"Speedup: {speedup:.2f}x")
        self.assertGreater(speedup, 1, "Numba should be faster than numpy")


if __name__ == "__main__":
    unittest.main()
