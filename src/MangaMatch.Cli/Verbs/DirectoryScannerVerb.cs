using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Drawing;
using System.Runtime.CompilerServices;

namespace MangaMatch.Cli.Verbs;

using ImageSignature;

[Verb("directory-scanner", HelpText = "Scan a directory for matching images.")]
internal class DirectoryScannerVerbOptions
{
    [Option('f', "file", HelpText = "The image scan for", Required = true)]
    public string File { get; set; } = string.Empty;

    [Option('d', "directory", HelpText = "The directory to scan", Required = true)]
    public string Directory { get; set; } = string.Empty;

    [Option('s', "subdirectories", HelpText = "Scan subdirectories", Default = false)]
    public bool SubDirectories { get; set; } = false;

    [Option('l', "lowes-ratio", HelpText = "The Lowes ratio to use", Default = 0.75f)]
    public float LowesRatio { get; set; } = 0.75f;

    [Option('h', "homography", HelpText = "Use homography to match images", Default = false)]
    public bool Homography { get; set; } = false;

    [Option('t', "features", HelpText = "The number of ORB features to detect", Default = 2000)]
    public int Features { get; set; } = 2000;
}

internal class DirectoryScannerVerb(
    ILogger<DirectoryScannerVerb> logger,
    IMatrixService _matrix) : BooleanVerb<DirectoryScannerVerbOptions>(logger)
{
    private Feature2D? _detector = null;

    public const int K = 2;
    public int Features { get; set; } = 2000;
    public float LowesRatio { get; set; } = 0.75f;
    public bool Homography { get; set; } = false;

    public Feature2D Detector => _detector ??= new ORB(Features);

    public string? GetProperFilePath(DirectoryScannerVerbOptions options)
    {
        if (string.IsNullOrEmpty(options.Directory))
        {
            _logger.LogWarning("No directory specified to scan in.");
            return null;
        }

        if (string.IsNullOrEmpty(options.File))
        {
            _logger.LogWarning("No file specified to scan for.");
            return null;
        }

        if (File.Exists(options.File))
            return options.File;

        var file = Path.Combine(options.Directory, options.File);
        if (File.Exists(file))
            return file;

        _logger.LogWarning("Cannot find file to scan for: {file}", options.File);
        return null;
    }

    public async Task<LoadedImage?> GetDescriptors(string file)
    {
        try
        {
            using var io = File.OpenRead(file);
            var image = await _matrix.GetMatrix(io);
            var points = new VectorOfKeyPoint();
            var descriptors = new Mat();

            Detector.DetectAndCompute(image, null, points, descriptors, false);
            return new(image, points, descriptors, file);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error occurred while loading file: {file}", file);
            return null;
        }
    }

    public Match MatchScoreBetter(LoadedImage model, LoadedImage observed, BFMatcher matcher, bool checkHomography)
    {
        try
        {
            var matches = new VectorOfVectorOfDMatch();
            matcher.KnnMatch(model.Descriptors, observed.Descriptors, matches, K);

            var matchingResults = new VectorOfDMatch(matches[0].ToArray());

            var mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));

            Features2DToolbox.VoteForUniqueness(matches, LowesRatio, mask);
            var nonZero = CvInvoke.CountNonZero(mask);

            if (nonZero < 4) return new(model, observed, new(), 0);

            nonZero = Features2DToolbox.VoteForSizeAndOrientation(model.Points, observed.Points, matches, mask, 1.5, 20);
            if (nonZero < 4) return new(model, observed, new(), 0);

            if (!checkHomography) return new(model, observed, matches[0], nonZero);

            var homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(
                model.Points, observed.Points, matches, mask, 1.5);
            if (homography is null || homography.IsEmpty)
                return new(model, observed, matches[0], 0);

            var nInliers = CvInvoke.CountNonZero(mask);
            return new(model, observed, matches[0], nInliers);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error occurred while comparing {first} to {second}", model.Path, observed.Path);
            return new(model, observed, new(), 0);
        }
    }

    public Match MatchScore(LoadedImage first, LoadedImage second, BFMatcher matcher, bool homography)
    {
        bool IsHomography2(VectorOfVectorOfDMatch matches)
        {
            var mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));

            var model = first.Points;
            var observed = second.Points;
            var homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(
                model, observed, matches, mask, 1.5);
            if (homography is null || homography.IsEmpty) return false;
            var nInliers = CvInvoke.CountNonZero(mask);
            return nInliers >= 63 && nInliers >= (observed.Size / 4);
        }

        bool HasHomography(List<MDMatch> goodMatches)
        {
            var points1 = new List<PointF>();
            var points2 = new List<PointF>();

            foreach (var match in goodMatches)
            {
                points1.Add(first.Points[match.QueryIdx].Point);
                points2.Add(second.Points[match.TrainIdx].Point);
            }

            if (points1.Count < 4 || points2.Count < 4) return false;

            var homography = CvInvoke.FindHomography([.. points2], [.. points1], RobustEstimationAlgorithm.Ransac, 2);
            return homography is not null && !homography.IsEmpty;
        }

        try
        {
            var matches = new VectorOfVectorOfDMatch();
            matcher.KnnMatch(first.Descriptors, second.Descriptors, matches, K);

            var goodMatches = new List<MDMatch>();
            for (var i = 0; i < matches.Size; i++)
            {
                if (matches[i].Size < 2) continue;

                var match = matches[i].ToArray();
                if (match[0].Distance < LowesRatio * match[1].Distance)
                    goodMatches.Add(match[0]);
            }

            var goodVector = new VectorOfDMatch(goodMatches.ToArray());

            if (homography && !HasHomography(goodMatches)) 
                return new(first, second, goodVector, 0);

            var count = (double)(first.Points.Size + second.Points.Size);
            return new(first, second, goodVector, goodMatches.Count * 2 / count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error occurred while comparing {first} to {second}", first.Path, second.Path);
            return new(first, second, new(), 0);
        }
    }

    public async IAsyncEnumerable<LoadedImage> ScanDirectory(string directory, bool tlo, 
        [EnumeratorCancellation]CancellationToken token)
    {
        string[] ext = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"];
        var search = tlo ? SearchOption.TopDirectoryOnly : SearchOption.AllDirectories;
        var files = Directory.GetFiles(directory, "*.*", search);

        foreach(var file in files)
        {
            if (token.IsCancellationRequested) yield break;

            if (!ext.Any(file.EndsWithIc)) continue;

            _logger.LogDebug("Starting to scan file: {path}", file);
            var match = await GetDescriptors(file);
            if (match is not null)
                yield return match;
        }
    }

    public void DrawMatch(string path, Match match)
    {
        if (File.Exists(path))
        {
            _logger.LogWarning("Match file already exists, deleting it: {path}", path);
            File.Delete(path);
        }

        using var result = new Mat();
        Features2DToolbox.DrawMatches(
            match.Second.Image, match.Second.Points, 
            match.First.Image, match.First.Points, 
            new VectorOfDMatch(match.Matching.ToArray()), 
            result, 
            new MCvScalar(0, 255, 0), new MCvScalar(255, 0, 0), 
            null, Features2DToolbox.KeypointDrawType.NotDrawSinglePoints);
        CvInvoke.Imwrite(path, result);
        _logger.LogDebug("Wrote match to file: {path}", path);
    }

    public override async Task<bool> Execute(DirectoryScannerVerbOptions options, CancellationToken token)
    {
        Features = options.Features;
        LowesRatio = options.LowesRatio;
        Homography = options.Homography;

        _logger.LogInformation("Detecting ORB features: {features}", Features);
        _logger.LogInformation("Lowes Ratio: {ratio}", LowesRatio);
        _logger.LogInformation("Using Homography: {homography}", Homography);
        _logger.LogInformation("Scanning directory: {directory}", options.Directory);
        _logger.LogInformation("Scanning for file: {file}", options.File);
        _logger.LogInformation("Scanning subdirectories: {subdirectories}", options.SubDirectories);

        var file = GetProperFilePath(options);
        if (file is null) return false;

        using var primary = await GetDescriptors(file);
        if (primary is null) return false;
        using var matcher = new BFMatcher(DistanceType.Hamming, false);
        //using var matcher = new BFMatcher(DistanceType.L2);
        //matcher.Add(primary.Descriptors);

        var matches = new List<Match>();
        await foreach(var match in ScanDirectory(options.Directory, !options.SubDirectories, token))
        {
            var result = MatchScore(primary, match, matcher, Homography);
            if (result.Score <= 0) 
            {
                _logger.LogWarning("Match score is 0, skipping: {path}", match.Path);
                match.Dispose();
                continue;
            }

            matches.Add(result);
            _logger.LogInformation("Found match: {path} with score {score}", match.Path, result.Score);
        }

        var topFive = matches.OrderByDescending(x => x.Score);
        int count = 0;
        foreach (var result in topFive)
        {
            if (count++ >= 5) break;

            _logger.LogInformation("Top LoadedImage: {path} with score {score}", result.Second.Path, result.Score);
            DrawMatch($"match-{count}.jpg", result);
        }

        foreach(var match in matches)
            match.Second.Dispose();

        return true;
    }

    public record class LoadedImage(Mat Image, VectorOfKeyPoint Points, Mat Descriptors, string Path) : IDisposable
    {
        public void Dispose()
        {
            Image.Dispose();
            Points.Dispose();
            Descriptors.Dispose();
            GC.SuppressFinalize(this);
        }
    }

    public record class Match(LoadedImage First, LoadedImage Second, VectorOfDMatch Matching, double Score);
}
