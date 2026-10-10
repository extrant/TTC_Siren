using Xunit;

namespace TTC.Siren.Tests;

/// <summary>Review M7-3: the golden-generation helper must not deadlock when the generator writes a lot to stdout and stderr.</summary>
public class GoldenHarnessTests
{
    // 1024 iterations x (1023 characters + newline) on each stream: at least 1 MiB to stdout and 1 MiB to stderr, interleaved.
    // Reading one pipe to the end before the other blocks the child as soon as the pipe the harness is not reading is full.
    private const string Writer =
        "import sys\n" +
        "for _ in range(1024):\n" +
        "    sys.stdout.write('o' * 1023 + '\\n')\n" +
        "    sys.stderr.write('e' * 1023 + '\\n')\n";

    [Fact]
    public async Task A_child_that_writes_a_megabyte_to_each_stream_completes_through_the_helper()
    {
        var run = Task.Run(() => GoldenFiles.RunPythonProcess(new[] { "-c", Writer }));
        var finished = await Task.WhenAny(run, Task.Delay(TimeSpan.FromSeconds(60)));
        Assert.True(ReferenceEquals(finished, run), "RunPythonProcess did not return: stdout/stderr are not read concurrently (M7-3)");
        var (exitCode, stdout, stderr) = await run;
        Assert.Equal(0, exitCode);
        Assert.True(stdout.Length >= 1024 * 1024, $"stdout {stdout.Length} characters");
        Assert.True(stderr.Length >= 1024 * 1024, $"stderr {stderr.Length} characters");
        Assert.Equal(1024, stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries).Length);
        Assert.Equal(1024, stderr.Split('\n', StringSplitOptions.RemoveEmptyEntries).Length);
    }

    [Fact]
    public void A_failing_child_reports_its_exit_code_and_both_streams()
    {
        var (exitCode, stdout, stderr) = GoldenFiles.RunPythonProcess(new[] { "-c", "import sys; print('to out'); print('to err', file=sys.stderr); sys.exit(3)" });
        Assert.Equal(3, exitCode);
        Assert.Contains("to out", stdout);
        Assert.Contains("to err", stderr);
    }
}
