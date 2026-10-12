// TTC_Siren_PyPy.exe 启动器
// PyInstaller 无法把 PyPy 打成单文件 exe（PyPy 的字节码/对象模型和 CPython 不兼容），
// 所以这个体积很小的原生 exe 只做一件事：以自己所在目录为基准，
// 调用同目录下 PYPY\pypy3.exe 运行 app.py，并把控制台、退出码原样转发。
//
// CPython 在 Windows 控制台下会自动走 WriteConsoleW（PEP 528），不受控制台代码页影响；
// PyPy 没有这个行为，写出的 UTF-8 字节会按当前代码页被控制台重新解读，
// 不设代码页就会出现方框字符/中文被显示成乱码的问题（run_pypy.bat 里的 chcp 65001
// 就是干这个的，这里用等价的 Win32 API 在启动子进程前设置好）。
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

class Launcher
{
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetConsoleOutputCP(uint wCodePageID);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetConsoleCP(uint wCodePageID);

    const uint CP_UTF8 = 65001;

    static int Main(string[] args)
    {
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);

        string baseDir = AppDomain.CurrentDomain.BaseDirectory;
        string pypyExe = Path.Combine(baseDir, "PYPY", "pypy3.exe");
        string appPy = Path.Combine(baseDir, "app.py");

        if (!File.Exists(pypyExe) || !File.Exists(appPy))
        {
            Console.WriteLine("安装不完整：找不到 PYPY\\pypy3.exe 或 app.py。");
            Console.WriteLine("请确认本 exe 与 PYPY 文件夹、app.py 等放在同一目录下。");
            Console.WriteLine("按任意键退出...");
            Console.ReadKey();
            return 1;
        }

        string quotedArgs = "";
        foreach (string a in args)
        {
            quotedArgs += " \"" + a.Replace("\"", "\\\"") + "\"";
        }

        var psi = new ProcessStartInfo
        {
            FileName = pypyExe,
            Arguments = "app.py" + quotedArgs,
            WorkingDirectory = baseDir,
            UseShellExecute = false,
        };
        // 双保险：控制台代码页之外，再让 Python 侧也强制按 UTF-8 编解码 stdio。
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";

        try
        {
            using (var proc = Process.Start(psi))
            {
                proc.WaitForExit();
                return proc.ExitCode;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("启动失败: " + ex.Message);
            Console.WriteLine("按任意键退出...");
            Console.ReadKey();
            return 1;
        }
    }
}
