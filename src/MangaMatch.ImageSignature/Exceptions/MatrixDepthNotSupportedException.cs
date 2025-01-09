using Emgu.CV;

namespace MangaMatch.ImageSignature.Exceptions;

/// <summary>
/// Thrown if the depth of the <see cref="Mat"/> is not supported
/// </summary>
public class MatrixDepthNotSupportedException() 
    : Exception("Unsupported Mat depth. Only CV_8U or CV_32F are supported.") { }
