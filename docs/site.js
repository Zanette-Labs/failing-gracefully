document.querySelectorAll("[data-copy]").forEach((button) => {
  button.hidden = false;
  let reset;
  button.addEventListener("click", async () => {
    const code = document.getElementById(button.dataset.copy);
    const status = document.getElementById("copy-status");
    clearTimeout(reset);
    try {
      await navigator.clipboard.writeText(code.textContent.trim());
      button.textContent = "Copied";
      status.textContent = "Citation copied to clipboard.";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = "Text selected";
      status.textContent = "Copy is unavailable. The citation is selected; use your browser's copy command.";
    }
    reset = setTimeout(() => { button.textContent = "Copy"; }, 2500);
  });
});

const contentsLinks = [...document.querySelectorAll(".outline a, .contents-inline a")];
const headings = [...document.querySelectorAll("main h2[id], main h3[id], main h4[id]")];
let scheduled = false;
let activeId = null;

function updateContents() {
  scheduled = false;
  let current = headings[0];
  for (const heading of headings) {
    if (heading.getBoundingClientRect().top <= 140) current = heading;
    else break;
  }
  if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) {
    current = headings[headings.length - 1];
  }
  if (!current || current.id === activeId) return;
  activeId = current.id;
  const sectionId = current.closest(".section")?.querySelector("h2[id]")?.id;
  for (const link of contentsLinks) {
    const active = link.hash === "#" + activeId;
    link.classList.toggle("within-section", !active && link.hash === "#" + sectionId);
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
}

function scheduleContentsUpdate() {
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(updateContents);
  }
}

window.addEventListener("scroll", scheduleContentsUpdate, { passive: true });
window.addEventListener("resize", scheduleContentsUpdate);
window.addEventListener("load", scheduleContentsUpdate);
document.fonts.ready.then(scheduleContentsUpdate);
updateContents();
