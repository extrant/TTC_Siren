using TTC.Siren.Server;
using TTC.Siren.Server.Security;

// TTC.Siren.Server: the product backend (Phase 8A). Serves POST /ai_move natively, with the engine in process.
//   dotnet TTC.Siren.Server.dll [--host 127.0.0.1] [--port 5000] [--workers N] [--queue 256] [--queue-wait-ms 60000]
//                               [--card-db path] [--tt-entries 4194304] [--memo-entries 32768] [--aging-ms 2000]
//   dotnet TTC.Siren.Server.dll keys add --name NAME | keys list | keys revoke ID | keys set-limit ID --rate N   (API keys)
if (args.Contains("--help") || args.Contains("-h"))
{
    Console.WriteLine("TTC.Siren.Server [--host 127.0.0.1] [--port 5000] [--workers N] [--queue 256] [--queue-wait-ms 60000]");
    Console.WriteLine("                 [--card-db path] [--tt-entries 4194304] [--memo-entries 32768]");
    Console.WriteLine("TTC.Siren.Server keys add --name NAME [--rate PER_MINUTE] [--burst N] [--inflight N] [--store PATH]   create an API key (shown once)");
    Console.WriteLine("TTC.Siren.Server keys list | keys revoke ID_OR_NAME | keys set-limit ID_OR_NAME [--rate N] [--burst N] [--inflight N]");
    Console.WriteLine("Settings may also be put in appsettings.json (section \"Server\") or environment variables (TTC_Server__Port ...).");
    return 0;
}

if (KeyCli.IsKeysCommand(args))
{
    try
    {
        return KeyCli.Run(args, ServerOptions.LoadSecurity(), Console.Out, Console.Error);
    }
    catch (ArgumentException e)
    {
        Console.Error.WriteLine("configuration error: " + e.Message);
        return 2;
    }
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
Console.WriteLine($"TTC.Siren.Server listening on {host.BaseUrl} {host.HttpsUrl}; workers={options.Workers}, queue={options.QueueLimit}, queue wait={options.QueueWaitMs} ms, " +
                  $"tt entries={options.TtEntries:N0}; card database sha256={host.CardDbSha256}");
await host.WaitForShutdownAsync();
return 0;
