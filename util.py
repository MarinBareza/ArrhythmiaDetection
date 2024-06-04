import numpy as np


def map_peak_annotations(peaks, annons):
    symbols = np.empty(0)
    for peak in peaks:
        absolute_differences = np.abs(peak - annons['sample'])
        closest_index = np.argmin(absolute_differences)
        symbols = np.append(symbols, annons[closest_index]['symbol'])

    peaks = peaks[:, np.newaxis]
    symbols = symbols[:, np.newaxis]
    peaks = np.concatenate((peaks, symbols), axis=1)
    return peaks
