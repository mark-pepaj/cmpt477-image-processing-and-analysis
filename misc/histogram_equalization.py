def histogram_equalization(img):
    p = get_intensity_probability(img) 
    cdf = np.zeros(256, dtype=float)

    for i in range(len(cdf)):
        for j in range(i+1):
            cdf[i] += p[j]

    cdf = np.round(255 * cdf).astype(np.uint8)

    return cdf[img.astype(np.uint8)]
