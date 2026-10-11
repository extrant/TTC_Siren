using System.Diagnostics;
using System.Text.Json;

namespace TTC.Siren.Tests;

/// <summary>
/// Review M8-1: the documented publish command (<c>dotnet publish csharp\src\TTC.Siren.Server -c Release -o &lt;dir&gt;</c>, docs/DEPLOY_WINDOWS.md section 2)
/// must work without any extra flag and must ship the SERVER's <c>appsettings.json</c>. It used to fail with NETSDK1152 because the Engine project, a library
/// of the server, also had a file of that name; with a flag that silenced the error either file might have won.
/// </summary>
public sealed class PublishTests : IDisposable
{
    private readonly string _out = Path.Combine(Path.GetTempPath(), "ttc-publish-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_out)) Directory.Delete(_out, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    /// <summary>Runs <c>dotnet &lt;arguments&gt;</c> in the repository root; both output streams are read concurrently (review M7-3).</summary>
    internal static (int ExitCode, string Output) RunDotnet(IReadOnlyList<string> arguments, int timeoutMs = 600_000)
    {
        var psi = new ProcessStartInfo("dotnet")
        {
            WorkingDirectory = GoldenFiles.RepoRoot, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false,
        };
        psi.Environment["DOTNET_CLI_UI_LANGUAGE"] = "en";
        psi.Environment["DOTNET_NOLOGO"] = "1";
        foreach (string a in arguments) psi.ArgumentList.Add(a);
        using var proc = Process.Start(psi) ?? throw new InvalidOperationException("could not start dotnet");
        var stdout = proc.StandardOutput.ReadToEndAsync();
        var stderr = proc.StandardError.ReadToEndAsync();
        if (!proc.WaitForExit(timeoutMs))
        {
            proc.Kill(entireProcessTree: true);
            throw new TimeoutException($"dotnet {string.Join(' ', arguments)} did not finish in {timeoutMs} ms");
        }
        Task.WaitAll(stdout, stderr);
        return (proc.ExitCode, stdout.Result + stderr.Result);
    }

    [Fact]
    public void The_documented_publish_command_works_without_flags_and_ships_the_servers_appsettings()
    {
        // exactly the command of docs/DEPLOY_WINDOWS.md section 2
        var (exit, output) = RunDotnet(new[] { "publish", Path.Combine("csharp", "src", "TTC.Siren.Server"), "-c", "Release", "-o", _out });
        Assert.True(exit == 0, "dotnet publish failed:\n" + output);
        Assert.DoesNotContain("NETSDK1152", output);

        // the configuration file is the server's: section "Server", port 5000, loopback; nothing of the engine
        using (var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(_out, "appsettings.json"))))
        {
            var root = doc.RootElement;
            Assert.True(root.TryGetProperty("Server", out var server), "appsettings.json has no \"Server\" section");
            Assert.Equal(5000, server.GetProperty("Port").GetInt32());
            Assert.Equal("127.0.0.1", server.GetProperty("Host").GetString());
            Assert.False(root.TryGetProperty("Engine", out _), "the engine's section ended up in the server's appsettings.json");
        }

        // the engine's host files are not shipped with the server
        Assert.False(File.Exists(Path.Combine(_out, "engine.appsettings.json")));
        Assert.False(File.Exists(Path.Combine(_out, "TTC.Siren.Engine.exe")));
        Assert.True(File.Exists(Path.Combine(_out, "TTC.Siren.Engine.dll")), "the engine library is part of the server");
        Assert.True(File.Exists(Path.Combine(_out, "TTC.Siren.Server.dll")));
        Assert.True(File.Exists(Path.Combine(_out, "TTC.Siren.Server.exe")));
        Assert.True(File.Exists(Path.Combine(_out, "cloudflare-ips.txt")), "the Cloudflare range file is published next to the program");

        // the GC choice of Phase 7 hardening (L7-1) is in the published runtime configuration
        using var runtime = JsonDocument.Parse(File.ReadAllText(Path.Combine(_out, "TTC.Siren.Server.runtimeconfig.json")));
        var configProperties = runtime.RootElement.GetProperty("runtimeOptions").GetProperty("configProperties");
        Assert.True(configProperties.GetProperty("System.GC.Server").GetBoolean());
        Assert.False(configProperties.GetProperty("System.GC.Concurrent").GetBoolean());
    }
}
