using TTC.Siren.Engine;

// TTC.Siren.Engine: the C# solver as a local HTTP service for the Python ai_server (Phase 7).
//   dotnet TTC.Siren.Engine.dll [--port 5300] [--workers N] [--queue 256] [--queue-wait-ms 60000]
//                               [--card-db path] [--tt-entries 4194304] [--memo-entries 32768] [--host 127.0.0.1] [--allow-remote]
if (args.Contains("--help") || args.Contains("-h"))
{
    Console.WriteLine("TTC.Siren.Engine [--host 127.0.0.1] [--port 5300] [--workers N] [--queue 256] [--queue-wait-ms 60000]");
    Console.WriteLine("                 [--card-db path] [--tt-entries 4194304] [--memo-entries 32768] [--allow-remote]");
    Console.WriteLine("Listens on a loopback address only unless --allow-remote is given. Settings may also be put in appsettings.json (section \"Engine\").");
    return 0;
}

EngineOptions options;
try
{
    options = EngineOptions.Parse(args);
}
catch (ArgumentException e)
{
    Console.Error.WriteLine("configuration error: " + e.Message);
    return 2;
}

var host = EngineHost.Create(options);
await host.StartAsync();
Console.WriteLine($"TTC.Siren.Engine protocol {Protocol.Version} listening on {host.BaseUrl}; workers={options.Workers}, queue={options.QueueLimit}, " +
                  $"queue wait={options.QueueWaitMs} ms, tt entries={options.TtEntries:N0}; card database sha256={host.CardDbSha256}");
await host.WaitForShutdownAsync();
return 0;
