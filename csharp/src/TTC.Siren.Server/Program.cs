using TTC.Siren.Server;

// TTC.Siren.Server: the product backend (Phase 8A). Serves POST /ai_move natively, with the engine in process.
//   dotnet TTC.Siren.Server.dll [--host 127.0.0.1] [--port 5000] [--workers N] [--queue 256] [--queue-wait-ms 60000]
//                               [--card-db path] [--tt-entries 4194304] [--memo-entries 32768]
if (args.Contains("--help") || args.Contains("-h"))
{
    Console.WriteLine("TTC.Siren.Server [--host 127.0.0.1] [--port 5000] [--workers N] [--queue 256] [--queue-wait-ms 60000]");
    Console.WriteLine("                 [--card-db path] [--tt-entries 4194304] [--memo-entries 32768]");
    Console.WriteLine("Settings may also be put in appsettings.json (section \"Server\") or environment variables (TTC_Server__Port ...).");
    return 0;
}

ServerOptions options;
try
{
    options = ServerOptions.Parse(args);
}
catch (ArgumentException e)
{
    Console.Error.WriteLine("configuration error: " + e.Message);
    return 2;
}

var host = ServerHost.Create(options);
await host.StartAsync();
Console.WriteLine($"TTC.Siren.Server listening on {host.BaseUrl}; workers={options.Workers}, queue={options.QueueLimit}, queue wait={options.QueueWaitMs} ms, " +
                  $"tt entries={options.TtEntries:N0}; card database sha256={host.CardDbSha256}");
await host.WaitForShutdownAsync();
return 0;
