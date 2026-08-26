using System.IO.Compression;

if (args.Length < 2)
{
    return 2;
}

var mode = args[0];
var archivePath = args[1];

using var archive = ZipFile.OpenRead(archivePath);

if (mode == "-Z1")
{
    foreach (var entry in archive.Entries)
    {
        Console.WriteLine(entry.FullName);
    }
    return 0;
}

if (mode == "-p" && args.Length >= 3)
{
    var entryName = args[2].Replace('\\', '/');
    var entry = archive.GetEntry(entryName);
    if (entry is null)
    {
        return 1;
    }

    await using var input = entry.Open();
    await using var output = Console.OpenStandardOutput();
    await input.CopyToAsync(output);
    return 0;
}

return 2;
