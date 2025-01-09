using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace MangaMatch.ImageSignature;

/// <summary>
/// A service for calculating whether or not a match is valid based on homography.
/// </summary>
public interface IHomographyService
{
    /// <summary>
    /// Calculate homography matrix from matched features and return true if the homography matrix is valid.
    /// </summary>
    /// <param name="matches">The vector of matches found</param>
    /// <param name="model">The key points of the image being searched</param>
    /// <param name="observed">The key points of the image in the collection</param>
    /// <param name="ransacThreshold">
    /// The maximum allowed reprojection error to treat a point pair as an inlier. If
    /// srcPoints and dstPoints are measured in pixels, it usually makes sense to set
    /// this parameter somewhere in the range 1 to 10.
    /// </param>
    /// <param name="minNInliers">The minimum number of inliers necessary to consider the match good</param>
    /// <param name="matchOffset">The denominator of the equation to consider a match good</param>
    /// <returns>Whether or not the homography matrix is valid</returns>
    bool CalculateHomography(VectorOfVectorOfDMatch matches, VectorOfKeyPoint model, VectorOfKeyPoint observed,
        double ransacThreshold = 1.5, int minNInliers = 63, int matchOffset = 4);

    /// <summary>
    /// Calculate homography matrix from the good matches 
    /// (from <see cref="CalculateGoodMatches(VectorOfVectorOfDMatch, double)"/>) 
    /// and return true if the homography matrix is valid.
    /// </summary>
    /// <param name="matches">The good matches found</param>
    /// <param name="model">The key points of the image being searched</param>
    /// <param name="observed">The key points of the image in the collection</param>
    /// <param name="ransacThreshold">
    /// The maximum allowed reprojection error to treat a point pair as an inlier. If
    /// srcPoints and dstPoints are measured in pixels, it usually makes sense to set
    /// this parameter somewhere in the range 1 to 10.
    /// </param>
    /// <returns>Whether or not the homography matrix is valid</returns>
    bool CalculateHomography(VectorOfDMatch matches, VectorOfKeyPoint model, VectorOfKeyPoint observed, double ransacThreshold = 2);

    /// <summary>
    /// Calculate the good match points based off of Lowe's distance ratio test.
    /// </summary>
    /// <param name="matches">The vector of matches found</param>
    /// <param name="lowesRatio">The lowes ratio to use during calculation</param>
    /// <returns>The good matches (if any)</returns>
    VectorOfDMatch? CalculateGoodMatches(VectorOfVectorOfDMatch matches, double lowesRatio = 1.5);
}

internal class HomographyService(
    ILogger<HomographyService> _logger) : IHomographyService
{
    public bool CalculateHomography(VectorOfVectorOfDMatch matches, VectorOfKeyPoint model, VectorOfKeyPoint observed, 
        double ransacThreshold = 1.5, int minNInliers = 63, int matchOffset = 4)
    {
        try
        {
            using var mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));

            using var homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(
                    model, observed, matches, mask, ransacThreshold);
            if (homography is null || homography.IsEmpty) return false;
            var nInliers = CvInvoke.CountNonZero(mask);
            return nInliers >= minNInliers && nInliers >= (observed.Size / matchOffset);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HomographyService] Error calculating homography matrix.");
            return false;
        }
    }

    public bool CalculateHomography(VectorOfDMatch matches, VectorOfKeyPoint model, VectorOfKeyPoint observed, double ransacThreshold = 2)
    {
        try
        {
            var points1 = new List<PointF>();
            var points2 = new List<PointF>();

            foreach (var match in matches.ToArray())
            {
                points1.Add(model[match.QueryIdx].Point);
                points2.Add(observed[match.TrainIdx].Point);
            }

            if (points1.Count < 4 || points2.Count < 4) return false;

            var homography = CvInvoke.FindHomography(
                [.. points2], [.. points1], RobustEstimationAlgorithm.Ransac, ransacThreshold);
            return homography is not null && !homography.IsEmpty;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HomographyService] Error calculating homography matrix.");
            return false;
        }
    }

    public VectorOfDMatch? CalculateGoodMatches(VectorOfVectorOfDMatch matches, double lowesRatio = 1.5)
    {
        try
        {
            var goodMatches = new List<MDMatch>();
            for (var i = 0; i < matches.Size; i++)
            {
                if (matches[i].Size < 2) continue;

                var match = matches[i].ToArray();
                if (match[0].Distance < lowesRatio * match[1].Distance)
                    goodMatches.Add(match[0]);
            }

            if (goodMatches.Count == 0) return null;

            return new VectorOfDMatch(goodMatches.ToArray());
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HomographyService] Error calculating good matches.");
            return null;
        }
    }
}
