# Accessibility

moex aims to be usable by everyone who can run it, including people who use a screen reader, magnification or the
keyboard only. This page says what works today, what doesn't, and how to tell us about a barrier.

## How moex is used

moex is a command-line program: everything happens as plain text in a terminal (Windows Terminal, PowerShell,
Command Prompt). There is no graphical window and no mouse is needed.

- **Plain text only.** moex prints no colors, no icons and no layout drawn with characters; information is never
  shown by color alone.
- **Keyboard only.** `moex chat` reads one line at a time: type, press Enter. `/reset` and `/exit` are typed commands.
- **Readable output.** Each reply ends with one summary line (`[60 tokens, 71.5 tok/s, ...]`), and errors say what
  went wrong and what to run next, in words.
- **Any text size and theme.** moex follows your terminal's font, size, zoom and colors.
- **Other apps.** `moex serve` gives an OpenAI- and Anthropic-compatible API, so you can use moex from any chat app
  or editor that works better with your assistive technology.
- **Documentation.** The README and these files are plain Markdown with headings, lists and tables, and the images
  in the README have text descriptions.

## Supported environments

Tested on Windows 10 and 11 in Windows Terminal, PowerShell and Command Prompt. Screen readers that read the terminal
(Narrator, NVDA, JAWS) should work with `moex chat` and `moex serve`, but they have not been tested systematically yet.

## Known limitations

- **Download progress.** `moex download` redraws its progress line in place several times a second; some screen
  readers announce every update. Workaround: start it and switch away, or run `moex download > progress.txt 2>&1`
  and check the file.
- **Streaming replies.** `moex chat` prints a reply word by word as it is written; some screen readers read it in
  small pieces. Waiting for the summary line, then reading the reply, works around it.
- **The speed chart** in the README is an image; its numbers are also in the Speed table below it.

## Report a barrier

If something in moex or its documentation is hard or impossible to use with your setup, please
[open an issue](https://github.com/neerajdad123-byte/moe/issues/new) and say what you tried, what happened, and which
assistive technology you use. Accessibility problems are treated as bugs.
