using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.Features2D;
using Emgu.CV.ImgHash;
using Emgu.CV.Util;

namespace MangaMatch.ImageSignature;

/// <summary>
/// A service for generating signatures for images
/// </summary>
public interface ISignatureService
{
    /// <summary>
    /// Generate the signatures for the given matrix using <see cref="CudaORBDetector"/>
    /// </summary>
    /// <param name="matrix">The matrix to process</param>
    /// <param name="keyPoints">The number of key points to detect</param>
    /// <param name="normalize">Whether or not to normalize the output vectors</param>
    /// <returns>The vectors of the image signatures</returns>
    /// <remarks>This should only be chosen if cuda support is available. Prefer <see cref="Orb(Mat, int, bool)"/> instead</remarks>
    float[][] OrbCuda(Mat matrix, int keyPoints, bool normalize);

    /// <summary>
    /// Generate the signatures for the given matrix using <see cref="ORB"/>
    /// </summary>
    /// <param name="matrix">The matrix to process</param>
    /// <param name="keyPoints">The number of key points to detect</param>
    /// <param name="normalize">Whether or not to normalize the output vectors</param>
    /// <returns>The vectors of the image signatures</returns>
    float[][] Orb(Mat matrix, int keyPoints, bool normalize);

    /// <summary>
    /// Generate the vectors for the given matrix using the given algorithm
    /// </summary>
    /// <param name="algorithm">The algorithm to use to generate the signatures</param>
    /// <param name="matrix">The matrix to process</param>
    /// <param name="normalize">Whether or not to normalize the output vectors</param>
    /// <returns>The vectors of the image signatures</returns>
    float[][] GenerateSignatures(Feature2D algorithm, Mat matrix, bool normalize);
}

internal class SignatureService(
    IMatrixService _matrix,
    ILogger<SignatureService> _logger) : ISignatureService
{
    public float[][] OrbCuda(Mat matrix, int keyPoints, bool normalize)
    {
        using var detector = new CudaORBDetector(keyPoints);
        return GenerateSignatures(detector, matrix, normalize);
    }

    public float[][] Orb(Mat matrix, int keyPoints, bool normalize)
    {
        using var detector = new ORB(keyPoints);
        return GenerateSignatures(detector, matrix, normalize);
    }

    public float[][] PHash(Mat matrix, bool normalize)
    {
        using var hash = new PHash();
        return GenerateHash(hash, matrix, normalize);
    }

    public float[][] GenerateHash(ImgHashBase algorithm, Mat matrix, bool normalize)
    {
        try
        {
            using var mat = new Mat();
            algorithm.Compute(matrix, mat);
            return _matrix.MatrixToVectors(mat, true).ToArray();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SignatureService] Error while generating hash: {error}", ex.Message);
            throw;
        }
    }

    public float[][] GenerateSignatures(Feature2D algorithm, Mat matrix, bool normalize)
    {
        try
        {
            using var keyPoints = new VectorOfKeyPoint();
            using var descriptors = new Mat();

            algorithm.DetectAndCompute(matrix, null, keyPoints, descriptors, false);
            return _matrix.MatrixToVectors(descriptors, normalize).ToArray();
        }
        catch (Exception e)
        {
            _logger.LogError(e, "[SignatureService] Error while generating descriptors: {error}", e.Message);
            throw;
        }
    }
}
