using Emgu.CV;
using Emgu.CV.CvEnum;

namespace MangaMatch.ImageSignature;

using Exceptions;

/// <summary>
/// A service for working with matrixes in OpenCV (EMGU)
/// </summary>
public interface IMatrixService
{
    /// <summary>
    /// Gets a matrix from the given stream
    /// </summary>
    /// <param name="stream">The stream to get the matrix from</param>
    /// <returns>The matrix that was fetched</returns>
    Task<Mat> GetMatrix(Stream stream);

    /// <summary>
    /// Gets a matrix from the given memory stream
    /// </summary>
    /// <param name="stream">The memory stream in question</param>
    /// <returns>The matrix that was fetched</returns>
    Mat GetMatrix(MemoryStream stream);

    /// <summary>
    /// Gets a matrix from the given byte array
    /// </summary>
    /// <param name="data">The data to create the matrix for</param>
    /// <returns>The matrix that was fetched</returns>
    Mat GetMatrix(byte[] data);

    /// <summary>
    /// Normalize a vector to be all positive numbers
    /// </summary>
    /// <param name="vector">The vectors to normalize</param>
    /// <returns>The normalized vector</returns>
    float[] NormalizeVector(float[] vector);

    /// <summary>
    /// Gets all of the descriptor vectors from the matrix
    /// </summary>
    /// <param name="mat">The matrix to pull from</param>
    /// <param name="normalize">Whether or not to normalize the vectors to positive numbers</param>
    /// <returns>All of the descriptors as vectors</returns>
    IEnumerable<float[]> MatrixToVectors(Mat mat, bool normalize);
}

internal class MatrixService(
    ILogger<MatrixService> _logger) : IMatrixService
{
    public Mat GetMatrix(byte[] data)
    {
        try
        {
            var mat = new Mat();
            CvInvoke.Imdecode(data, ImreadModes.Grayscale, mat);
            if (mat.IsEmpty)
                throw new MatrixEmptyException();

            return mat;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MatrixService] Error while decoding matrix: {error}", ex.Message);
            throw;
        }
    }

    public Mat GetMatrix(MemoryStream stream)
    {
        return GetMatrix(stream.ToArray());
    }

    public async Task<Mat> GetMatrix(Stream stream)
    {
        using var ms = new MemoryStream();
        await stream.CopyToAsync(ms);

        return GetMatrix(ms);
    }

    public float[] NormalizeVector(float[] vector)
    {
        float magnitude = (float)Math.Sqrt(vector.Sum(x => x * x));
        return vector.Select(x => x / magnitude).ToArray();
    }

    public IEnumerable<float[]> MatrixToVectors(Mat mat, bool normalize)
    {
        int typeSize = mat.ElementSize;

        // Ensure the type of the matrix is compatible (e.g., CV_8U or CV_32F)
        if (mat.Depth != DepthType.Cv8U && mat.Depth != DepthType.Cv32F)
            throw new MatrixDepthNotSupportedException();

        // Copy the raw data to a managed buffer
        byte[] buffer = new byte[mat.Rows * mat.Cols * typeSize];
        mat.CopyTo(buffer);
        //Marshal.Copy(mat.DataPointer, buffer, 0, buffer.Length);

        // Convert the buffer to float[][] based on the matrix type
        for (int i = 0; i < mat.Rows; i++)
        {
            var current = new float[mat.Cols];
            for (int j = 0; j < mat.Cols; j++)
            {
                if (mat.Depth == DepthType.Cv8U)
                {
                    // For CV_8U, scale to a float range if needed
                    current[j] = buffer[i * mat.Cols + j];
                    continue;
                }
                // For CV_32F, directly cast to float
                current[j] = BitConverter.ToSingle(buffer, (i * mat.Cols + j) * typeSize);
            }

            if (normalize)
                current = NormalizeVector(current);

            yield return current;
        }
    }
}
