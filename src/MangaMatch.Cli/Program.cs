using MangaMatch.Cli.Verbs;
using MangaMatch.ImageSignature;


return await new ServiceCollection()
    .AddSerilog()
    .AddSignaturing()
    .Cli(args, c => c
        .Add<TestSignaturesVerb>()
        .Add<DirectoryScannerVerb>());