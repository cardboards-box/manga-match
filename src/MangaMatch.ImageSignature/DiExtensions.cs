namespace MangaMatch.ImageSignature;

/// <summary>
/// Extensions for adding image signaturing to the DI container
/// </summary>
public static class DiExtensions
{
    /// <summary>
    /// Adds the image signaturing services to the DI container
    /// </summary>
    /// <param name="services">The service collection to add to</param>
    /// <returns>The service collection for fluent method chaining</returns>
    public static IServiceCollection AddSignaturing(this IServiceCollection services)
    {
        return services
            .AddTransient<IMatrixService, MatrixService>()
            .AddTransient<ISignatureService, SignatureService>();
    }
}
