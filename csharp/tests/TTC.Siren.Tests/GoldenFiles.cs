using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using TTC.Siren.Core;

namespace TTC.Siren.Tests;

/// <summary>Locates the repository and (re)generates Python golden data on demand. Never skips: failure to generate fails the test.</summary>
internal static class GoldenFiles
{
    public const int RulesTraceSeed = 20261007;
    public const int RulesTraceCount = 20000;

    /// <summary>Must equal GENERATOR_VERSION in tools/parity/py_rules_trace.py; bump both when Python semantics change.</summary>
    public const string GeneratorVersion = "py_rules_trace/4";

    private static readonly object Gate = new();

    public static string RepoRoot { get; } = FindRepoRoot();

    private static string FindRepoRoot()
    {
        string dataFile = CardDb.LocateDataFile(AppContext.BaseDirectory);
        return Path.GetDirectoryName(Path.GetDirectoryName(dataFile))!;
    }

    public static string GoldenDir => Path.Combine(RepoRoot, "tools", "parity", "golden");

    public static string RulesTracePath(int seed = RulesTraceSeed, int count = RulesTraceCount) =>
        Path.Combine(GoldenDir, $"rules_trace_g4_s{seed}_n{count}.jsonl.gz");

    public static string CardDbPath => Path.Combine(GoldenDir, "carddb.json");

    public static string EnsureRulesTrace(int seed = RulesTraceSeed, int count = RulesTraceCount)
    {
        string path = RulesTracePath(seed, count);
        Ensure(path, $"--out \"{{0}}\" --seed {seed} --count {count}", kind: "rules");
        return path;
    }

    public static string EnsureCardDb()
    {
        Ensure(CardDbPath, "--carddb-out \"{0}\"", kind: "carddb");
        return CardDbPath;
    }

    public const int SearchSeed = 20261008;
    public const string SearchGeneratorVersion = "py_search_trace/3";

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
        string path = Path.Combine(GoldenDir, $"search_{kind}_g3_s{SearchSeed}_n{count}.jsonl.gz");
        Ensure(path, $"--kind {kind} --out \"{{0}}\" --seed {SearchSeed} --count {count}", "py_search_trace.py", "search");
        return path;
    }

    public const int EndgameSeed = 20261009;
    public const string EndgameGeneratorVersion = "py_endgame_trace/2";

    /// <summary>Number of records per kind of tools/parity/py_endgame_trace.py.</summary>
    public static int EndgameCount(string kind) => kind switch
    {
        "scenario" => 500,
        "unique" => 800,
        "corner" => 1_500,
        "reply" => 1_500,
        "exact" => 700,
        "robust" => 400,
        "override" => 6_000,
        _ => throw new ArgumentException(kind),
    };

    public static string EnsureEndgameGolden(string kind)
    {
        int count = EndgameCount(kind);
        string path = Path.Combine(GoldenDir, $"endgame_{kind}_g2_s{EndgameSeed}_n{count}.jsonl.gz");
        Ensure(path, $"--kind {kind} --out \"{{0}}\" --seed {EndgameSeed} --count {count}", "py_endgame_trace.py", "endgame");
        return path;
    }

    // ------------------------------------------------------------------ Phase 7 end-to-end reference

    public const int E2eSeed = 20261010;
    public const int E2eCount = 360;
    public const string E2eGeneratorVersion = "phase7_e2e_reference/1";

    public static string E2eCorpusPath => Path.Combine(RepoRoot, "tools", "phase7", "corpus", $"corpus_s{E2eSeed}_n{E2eCount}.json");

    /// <summary>
    /// The Python reference of the end-to-end comparison (tools/phase7/e2e_reference.py): per corpus request what Python computes
    /// and what the Python server sends to the engine. Generated on demand (several minutes) and bound to the Python source.
    /// </summary>
    public static string EnsureE2eReference()
    {
        string path = Path.Combine(GoldenDir, $"e2e_reference_g1_s{E2eSeed}_n{E2eCount}.jsonl.gz");
        lock (Gate)
        {
            if (File.Exists(path) && StaleReason(path, "e2e") is null) return path;
        }
        // the corpus first (deterministic, a second), then the reference that reads it
        Directory.CreateDirectory(Path.GetDirectoryName(E2eCorpusPath)!);
        RunPython("tools/phase7/gen_corpus.py", "--out", E2eCorpusPath, "--seed", E2eSeed.ToString(), "--count", E2eCount.ToString());
        Ensure(path, "--corpus @@CORPUS@@ --out {0}", "tools/phase7/e2e_reference.py", "e2e");
        return path;
    }

    private static void RunPython(string script, params string[] args)
    {
        lock (Gate)
        {
            var arguments = new List<string> { script };
            arguments.AddRange(args);
            var (exitCode, stdout, stderr) = RunPythonProcess(arguments);
            if (exitCode != 0) throw new InvalidOperationException($"{script} failed (exit {exitCode}):{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        }
    }

    /// <summary>
    /// Runs <c>python &lt;arguments&gt;</c> in the repository root and returns its exit code and everything it wrote. Review M7-3: stdout
    /// and stderr are read concurrently (both asynchronously, then <c>WaitForExit</c>). Reading one stream to the end before the other
    /// deadlocks as soon as the child fills the pipe the harness is not reading: a generator that logs a lot to stderr hung the
    /// regeneration of the e2e golden for 35 minutes.
    /// </summary>
    internal static (int ExitCode, string StdOut, string StdErr) RunPythonProcess(IReadOnlyList<string> arguments)
    {
        var psi = new ProcessStartInfo("python")
        {
            WorkingDirectory = RepoRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        psi.Environment["PYTHONIOENCODING"] = "utf-8";
        foreach (string a in arguments) psi.ArgumentList.Add(a);
        using var proc = Process.Start(psi) ?? throw new InvalidOperationException("Could not start python");
        var stdout = proc.StandardOutput.ReadToEndAsync();
        var stderr = proc.StandardError.ReadToEndAsync();
        proc.WaitForExit();
        Task.WaitAll(stdout, stderr);
        return (proc.ExitCode, stdout.Result, stderr.Result);
    }

    /// <summary>
    /// Review M4-2: the header of a golden file binds it to the Python source (SHA-256 of every file it depends on and
    /// the Python major.minor version). Null if the file matches the working tree, otherwise the reason it is stale.
    /// </summary>
    public static string? StaleReason(string path, string kind)
    {
        JsonElement header;
        using (var file = File.OpenRead(path))
        {
            if (path.EndsWith(".gz", StringComparison.Ordinal))
            {
                using var gz = new GZipStream(file, CompressionMode.Decompress);
                using var reader = new StreamReader(gz);
                using var doc = JsonDocument.Parse(reader.ReadLine()!);
                header = doc.RootElement.GetProperty("header").Clone();
            }
            else
            {
                using var doc = JsonDocument.Parse(file);
                if (!doc.RootElement.TryGetProperty("header", out var h)) return "no header (golden predates source binding)";
                header = h.Clone();
            }
        }
        return GoldenSources.Check(header, RepoRoot, kind, GoldenSources.PythonVersion);
    }

    private static void Ensure(string path, string argsTemplate, string script = "py_rules_trace.py", string kind = "rules")
    {
        lock (Gate)
        {
            if (File.Exists(path))
            {
                string? stale = StaleReason(path, kind);
                if (stale is null) return;
                // A stale golden file must never be used: it was generated from different Python source. Regenerate.
                Console.Error.WriteLine($"golden {Path.GetFileName(path)} is stale ({stale}); regenerating");
                File.Delete(path);
            }
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            string tmp = path + ".tmp";
            if (File.Exists(tmp)) File.Delete(tmp);

            // Split the template first so a path containing spaces stays a single argument.
            var arguments = new List<string> { script.Contains('/') ? script : Path.Combine("tools", "parity", script) };
            foreach (string part in argsTemplate.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                arguments.Add(part.Contains("{0}") ? tmp : part == "@@CORPUS@@" ? E2eCorpusPath : part);

            var (exitCode, stdout, stderr) = RunPythonProcess(arguments);
            if (exitCode != 0 || !File.Exists(tmp))
                throw new InvalidOperationException($"Golden generation failed (exit {exitCode}):\n{stdout}\n{stderr}");
            File.Move(tmp, path);
        }
    }
}
