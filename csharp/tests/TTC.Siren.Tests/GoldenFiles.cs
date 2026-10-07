using System.Diagnostics;
using TTC.Siren.Core;

namespace TTC.Siren.Tests;

/// <summary>Locates the repository and (re)generates Python golden data on demand. Never skips: failure to generate fails the test.</summary>
internal static class GoldenFiles
{
    public const int RulesTraceSeed = 20261007;
    public const int RulesTraceCount = 20000;

    /// <summary>Must equal GENERATOR_VERSION in tools/parity/py_rules_trace.py; bump both when Python semantics change.</summary>
    public const string GeneratorVersion = "py_rules_trace/2";

    private static readonly object Gate = new();

    public static string RepoRoot { get; } = FindRepoRoot();

    private static string FindRepoRoot()
    {
        string dataFile = CardDb.LocateDataFile(AppContext.BaseDirectory);
        return Path.GetDirectoryName(Path.GetDirectoryName(dataFile))!;
    }

    public static string GoldenDir => Path.Combine(RepoRoot, "tools", "parity", "golden");

    public static string RulesTracePath(int seed = RulesTraceSeed, int count = RulesTraceCount) =>
        Path.Combine(GoldenDir, $"rules_trace_g2_s{seed}_n{count}.jsonl.gz");

    public static string CardDbPath => Path.Combine(GoldenDir, "carddb.json");

    public static string EnsureRulesTrace(int seed = RulesTraceSeed, int count = RulesTraceCount)
    {
        string path = RulesTracePath(seed, count);
        Ensure(path, $"--out \"{{0}}\" --seed {seed} --count {count}");
        return path;
    }

    public static string EnsureCardDb()
    {
        Ensure(CardDbPath, "--carddb-out \"{0}\"");
        return CardDbPath;
    }

    public const int SearchSeed = 20261008;
    public const string SearchGeneratorVersion = "py_search_trace/1";

    /// <summary>Record counts per kind of tools/parity/py_search_trace.py (eval: positions; search/id: positions; tt: families).</summary>
    public static int SearchCount(string kind) => kind switch
    {
        "eval" => 100_000,
        "search" => 900,
        "id" => 200,
        "tt" => 3_000,
        _ => throw new ArgumentException(kind),
    };

    public static string EnsureSearchGolden(string kind)
    {
        int count = SearchCount(kind);
        string path = Path.Combine(GoldenDir, $"search_{kind}_g1_s{SearchSeed}_n{count}.jsonl.gz");
        Ensure(path, $"--kind {kind} --out \"{{0}}\" --seed {SearchSeed} --count {count}", "py_search_trace.py");
        return path;
    }

    private static void Ensure(string path, string argsTemplate, string script = "py_rules_trace.py")
    {
        lock (Gate)
        {
            if (File.Exists(path)) return;
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            string tmp = path + ".tmp";
            if (File.Exists(tmp)) File.Delete(tmp);

            var psi = new ProcessStartInfo("python")
            {
                WorkingDirectory = RepoRoot,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            psi.Environment["PYTHONIOENCODING"] = "utf-8";
            // Split the template first so a path containing spaces stays a single argument.
            psi.ArgumentList.Add(Path.Combine("tools", "parity", script));
            foreach (string part in argsTemplate.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                psi.ArgumentList.Add(part.Contains("{0}") ? tmp : part);

            using var proc = Process.Start(psi) ?? throw new InvalidOperationException("Could not start python");
            string stdout = proc.StandardOutput.ReadToEnd();
            string stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit();
            if (proc.ExitCode != 0 || !File.Exists(tmp))
                throw new InvalidOperationException($"Golden generation failed (exit {proc.ExitCode}):\n{stdout}\n{stderr}");
            File.Move(tmp, path);
        }
    }
}
