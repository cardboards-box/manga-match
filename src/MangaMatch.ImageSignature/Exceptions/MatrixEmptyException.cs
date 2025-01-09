using Emgu.CV;

namespace MangaMatch.ImageSignature.Exceptions;

/// <summary>
/// Thrown when a <see cref="Mat"/> is empty
/// </summary>
public class MatrixEmptyException() 
    : Exception("The given matrix is empty.") { }
