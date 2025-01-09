namespace MangaMatch.Cli.Verbs;

using ImageSignature;

[Verb("test-signatures", HelpText = "Test the signature generation.")]
internal class TestSignaturesVerbOptions
{
    [Option('f', "file", HelpText = "The file to generate the signature for.")]
    public string? File { get; set; }
}

//0.6 - 1000 (OK)
//0.6 - 1500 (DECENT)
internal class TestSignaturesVerb(
    ILogger<TestSignaturesVerb> logger,
    IMatrixService _matrix,
    ISignatureService _signatures) : BooleanVerb<TestSignaturesVerbOptions>(logger)
{
    private const string DEFAULT_IMAGE = @"F:\Pictures\downloads\4-f359cb44a760571a0fd7ae22020eaeae994f73761ebc2a2339b3f1aad4ab62dc.png";

    public override async Task<bool> Execute(TestSignaturesVerbOptions options, CancellationToken token)
    {
        var file = options.File.ForceNull() ?? DEFAULT_IMAGE;
        if (!File.Exists(file))
        {
            _logger.LogWarning("Could not find file to process: {file}", file);
            return false;
        }

        using var io = File.OpenRead(file);
        using var matrix = await _matrix.GetMatrix(io);
        var signatures = _signatures.Orb(matrix, 5000, true);

        var data = JsonSerializer.Serialize(signatures);
        await File.WriteAllTextAsync("signatures.json", data, token);

        var total = signatures.LongLength;
        var memory = total * sizeof(float);
        var size = FormatBytes(memory);

        var eightMillionOfThem = memory * 8_000_000;
        var size2 = FormatBytes(eightMillionOfThem);

        _logger.LogInformation("Signature Written: {length} - Single Signature: {memory} - 8 million of them: {size2}", signatures.LongLength, size, size2);
        return true;
    }

    private static string FormatBytes(long bytes)
    {
        string[] Suffix = ["B", "KB", "MB", "GB", "TB"];
        int i;
        double dblSByte = bytes;
        for (i = 0; i < Suffix.Length && bytes >= 1024; i++, bytes /= 1024)
            dblSByte = bytes / 1024.0;

        return string.Format("{0:0.##} {1}", dblSByte, Suffix[i]);
    }
}
