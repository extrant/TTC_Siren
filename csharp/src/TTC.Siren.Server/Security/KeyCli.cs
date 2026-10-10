using System.Globalization;

namespace TTC.Siren.Server.Security;

/// <summary>
/// The admin commands of the server binary (decision A-21):
/// <code>
/// TTC.Siren.Server keys add --name NAME [--rate PER_MINUTE] [--burst N] [--inflight N]   creates a key and prints it ONCE
/// TTC.Siren.Server keys list                                                               id, name, state, limits (never a secret)
/// TTC.Siren.Server keys revoke ID_OR_NAME
/// TTC.Siren.Server keys set-limit ID_OR_NAME [--rate PER_MINUTE] [--burst N] [--inflight N]
/// </code>
/// All accept <c>--store PATH</c> (default: the configured key store file). The running server picks up changes within a second.
/// </summary>
public static class KeyCli
{
    public static bool IsKeysCommand(string[] args) => args.Length > 0 && args[0] == "keys";

    public static int Run(string[] args, SecurityOptions security, TextWriter output, TextWriter error)
    {
        if (args.Length < 2)
        {
            error.WriteLine("usage: keys add|list|revoke|set-limit ...  (see --help)");
            return 2;
        }
        string command = args[1];
        var named = new Dictionary<string, string>(StringComparer.Ordinal);
        var positional = new List<string>();
        for (int i = 2; i < args.Length; i++)
        {
            if (args[i].StartsWith("--", StringComparison.Ordinal))
            {
                if (i + 1 >= args.Length)
                {
                    error.WriteLine($"{args[i]} needs a value");
                    return 2;
                }
                named[args[i]] = args[++i];
            }
            else positional.Add(args[i]);
        }

        string path = named.TryGetValue("--store", out var p) ? p : security.KeyStoreFile;
        var store = new ApiKeyStore(path);
        try
        {
            int? Int(string name) => named.TryGetValue(name, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : null;
            switch (command)
            {
                case "add":
                {
                    if (!named.TryGetValue("--name", out var name))
                    {
                        error.WriteLine("keys add needs --name");
                        return 2;
                    }
                    var (record, secret) = store.Add(name, Int("--rate") ?? security.DefaultRatePerMinute, Int("--burst") ?? security.DefaultBurst, Int("--inflight") ?? security.DefaultMaxInFlight);
                    output.WriteLine($"created key {record.Id} for '{record.Name}' ({record.RatePerMinute}/min, burst {record.Burst}, {record.MaxInFlight} in flight)");
                    output.WriteLine("The key is shown ONCE and cannot be recovered; hand it to the user over a safe channel:");
                    output.WriteLine(secret);
                    return 0;
                }
                case "list":
                    output.WriteLine("id          name                  state     rate/min  burst  in-flight  created               hash");
                    foreach (var r in store.List())
                        output.WriteLine($"{r.Id,-11} {r.Name,-21} {(r.Enabled ? "enabled" : "REVOKED"),-9} {r.RatePerMinute,8}  {r.Burst,5}  {r.MaxInFlight,9}  {r.Created:yyyy-MM-dd HH:mm}   {r.Hash[..12]}…");
                    return 0;
                case "revoke":
                    if (positional.Count != 1)
                    {
                        error.WriteLine("keys revoke needs the id or the name");
                        return 2;
                    }
                    if (!store.Revoke(positional[0]))
                    {
                        error.WriteLine("no such key");
                        return 1;
                    }
                    output.WriteLine("revoked " + positional[0]);
                    return 0;
                case "set-limit":
                    if (positional.Count != 1)
                    {
                        error.WriteLine("keys set-limit needs the id or the name");
                        return 2;
                    }
                    if (!store.SetLimit(positional[0], Int("--rate"), Int("--burst"), Int("--inflight")))
                    {
                        error.WriteLine("no such key");
                        return 1;
                    }
                    output.WriteLine("updated " + positional[0]);
                    return 0;
                default:
                    error.WriteLine("unknown keys command: " + command);
                    return 2;
            }
        }
        catch (Exception e) when (e is ArgumentException or InvalidOperationException or FormatException)
        {
            error.WriteLine(e.Message);
            return 2;
        }
    }
}
