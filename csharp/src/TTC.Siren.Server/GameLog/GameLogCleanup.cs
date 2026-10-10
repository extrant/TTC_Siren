using Microsoft.Extensions.Hosting;

namespace TTC.Siren.Server.GameLog;

/// <summary>The retention job: removes game logs older than the configured number of days, at start-up and then once a day.</summary>
public sealed class GameLogCleanup : BackgroundService
{
    private readonly GameLogStore _store;

    public GameLogCleanup(GameLogStore store) => _store = store;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                _store.Cleanup();
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                Console.Error.WriteLine("game log cleanup failed: " + e.Message);
            }
            try
            {
                await Task.Delay(TimeSpan.FromHours(24), stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }
    }
}
