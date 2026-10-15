// Browser-rendered SVG version of the opening visual storyboard.
// Simplex trajectories come from the same sampled optimization as Figure 2.
// AIME points are the paper's gracefulrl.csv at 200, 400, and 800 steps.
import { simplexData } from "./simplex-data.js";

const svg = document.getElementById("gracefulrl-storyboard");
const controls = document.querySelector(".teaser-controls");
const playPause = document.getElementById("storyboard-play-pause");
const ns = "http://www.w3.org/2000/svg";
const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
const narrow = window.matchMedia("(max-width: 720px)");
const simplexFrameDuration = 100;
const laterFrameDuration = 70;
const firstHoldFrame = 28;
const finalFrame = 82;
const firstHoldStart = firstHoldFrame * simplexFrameDuration;
const animationDuration = firstHoldStart + (finalFrame - firstHoldFrame) * laterFrameDuration;
const firstHoldDuration = 3000;
const endHoldDuration = 3000;
const cycleDuration = animationDuration + firstHoldDuration + endHoldDuration;
const ink = "#302820";
const muted = "#716459";
const panel = "#fff";
const rule = "#ded3c5";
const blue = "#4c78a8";
const orange = "#b66115";
const green = "#55a24d";
const methods = [
  { key: "reinforce", title: "REINFORCE", color: blue },
  { key: "graceful", title: "GracefulRL", color: orange },
];
// [error@1 (%), success@1 (%)], indexed by caution level.
const frontier = {
  200: { 0: [73.6458, 16.3889], 0.2: [27.1875, 15.7986], 0.4: [26.5972, 15.4861], 0.6: [21.1111, 14.6528], 0.8: [18.9583, 14.4097] },
  400: { 0: [80.9028, 18.1597], 0.2: [37.6736, 17.7778], 0.4: [20.4514, 16.9097], 0.6: [12.0486, 16.8403], 0.8: [8.2639, 15.625] },
  800: { 0: [79.0625, 20.1389], 0.2: [52.1528, 19.8958], 0.4: [23.2292, 19.0278], 0.6: [13.8194, 18.0903], 0.8: [8.8542, 16.9792] },
};
const cautions = [0.8, 0.6, 0.4, 0.2, 0];

function node(tag, attributes = {}, value) {
  const element = document.createElementNS(ns, tag);
  for (const [key, item] of Object.entries(attributes)) element.setAttribute(key, String(item));
  if (value !== undefined) element.textContent = value;
  return element;
}
function group(parent, transform) {
  const element = node("g", transform ? { transform } : {});
  parent.append(element);
  return element;
}
function line(parent, x1, y1, x2, y2, stroke, width = 1, extra = {}) {
  const element = node("line", { x1, y1, x2, y2, stroke, "stroke-width": width, ...extra });
  parent.append(element);
  return element;
}
function label(parent, x, y, value, size, fill = ink, extra = {}) {
  const element = node("text", { x, y, fill, "font-size": size, ...extra }, value);
  parent.append(element);
  return element;
}
function circle(parent, x, y, radius, fill, stroke = "none", width = 0) {
  const element = node("circle", { cx: x, cy: y, r: radius, fill, stroke, "stroke-width": width });
  parent.append(element);
  return element;
}
function ease(value) {
  const t = Math.max(0, Math.min(1, value));
  return t * t * (3 - 2 * t);
}
function mix(a, b, t) { return a + (b - a) * t; }
function position([success, fallback, error]) {
  return [220 * success + 90 * fallback + 350 * error, 112 * success + 295 * (fallback + error)];
}
function formatPoint([x, y]) { return `${x.toFixed(2)},${y.toFixed(2)}`; }

function probability(parent, x, y, outcome) {
  const text = label(parent, x, y, "", 14, muted, { "text-anchor": "middle" });
  text.append(node("tspan", { "font-style": "italic" }, "p"));
  text.append(node("tspan", { "baseline-shift": "sub", "font-size": 10 }, outcome));
  text.append(node("tspan", {}, " = 1"));
}

function simplexPanel(parent, method) {
  const card = group(parent);
  label(card, 220, 30, method.title, 23, method.color, { "text-anchor": "middle", "font-weight": 600, class: "storyboard-display" });
  card.append(node("path", { d: "M220 112 L90 295 L350 295 Z", fill: "#f8f5ef", stroke: "#b6a18a", "stroke-width": 1.5 }));
  for (const fraction of [0.25, 0.5, 0.75]) {
    for (const [a, b] of [
      [[fraction, 1 - fraction, 0], [fraction, 0, 1 - fraction]],
      [[1 - fraction, fraction, 0], [0, fraction, 1 - fraction]],
      [[1 - fraction, 0, fraction], [0, 1 - fraction, fraction]],
    ]) {
      const start = position(a);
      const end = position(b);
      line(card, ...start, ...end, rule, 0.8);
    }
  }
  const histories = simplexData.trajectories[method.key];
  const trails = histories.map((history) => {
    const points = history.map(position);
    const trail = node("polyline", { fill: "none", stroke: method.color, "stroke-width": 2, "stroke-opacity": 0.64, "stroke-linejoin": "round", "stroke-linecap": "round" });
    card.append(trail);
    circle(card, ...points[0], 3.4, panel, muted, 1.2);
    const current = circle(card, ...points[0], 5, method.color, panel, 1.5);
    return { points, trail, current };
  });
  for (const [name, outcome, x, nameY, probabilityY] of [
    ["Success", "S", 220, 63, 84],
    ["Fallback", "F", 90, 318, 340],
    ["Error", "E", 350, 318, 340],
  ]) {
    label(card, x, nameY, name, 15, ink, { "text-anchor": "middle", "font-weight": 500 });
    probability(card, x, probabilityY, outcome);
  }
  line(card, 16, 350, 424, 350, rule);
  label(card, 18, 378, "Mean reward", 16, muted);
  const reward = label(card, 422, 378, "", 20, method.color, { "text-anchor": "end", "font-weight": 700 });
  return { histories, trails, reward };
}

function desktopLegend(parent) {
  const g = group(parent, "translate(0 404)");
  line(g, 35, 0, 865, 0, rule, 1);
  label(g, 450, 35, "3-outcome softmax policy", 21, muted, { "text-anchor": "middle" });
  const first = label(g, 450, 80, "", 22, ink, { "text-anchor": "middle" });
  first.append(node("tspan", { fill: blue, class: "storyboard-display" }, "REINFORCE"));
  first.append(node("tspan", {}, " has vanishing gradients at suboptimal outcomes."));
  const second = label(g, 450, 113, "", 22, ink, { "text-anchor": "middle" });
  second.append(node("tspan", { fill: orange, class: "storyboard-display" }, "GracefulRL"));
  second.append(node("tspan", {}, " allows optimization to succeed."));
}

function mobileLegend(parent) {
  const g = group(parent, "translate(20 640)");
  line(g, 10, 0, 330, 0, rule, 1);
  label(g, 170, 30, "3-outcome softmax policy", 14.5, muted, { "text-anchor": "middle" });
  label(g, 170, 68, "REINFORCE: gradients vanish near fallback.", 15, blue, { "text-anchor": "middle", class: "storyboard-display" });
  label(g, 170, 103, "GracefulRL: optimization succeeds.", 16, orange, { "text-anchor": "middle", class: "storyboard-display" });
}

function frontierAt(step) {
  const low = step <= 400 ? 200 : 400;
  const high = step <= 400 ? 400 : 800;
  const t = (step - low) / (high - low);
  return Object.fromEntries(cautions.map((c) => [c, [
    mix(frontier[low][c][0], frontier[high][c][0], t),
    mix(frontier[low][c][1], frontier[high][c][1], t),
  ]]));
}

function frontierScene(parent, mobile) {
  const w = mobile ? 340 : 560;
  const x0 = mobile ? 50 : 81;
  const x1 = mobile ? 318 : 518;
  const y0 = mobile ? 149 : 135;
  const y1 = mobile ? 388 : 482;
  const g = group(parent);
  label(g, mobile ? 17 : 24, mobile ? 30 : 39, "Reliability Frontier", mobile ? 20 : 23, ink, { "font-weight": 600, class: "storyboard-display" });
  const compute = label(g, mobile ? 17 : w - 25, mobile ? 55 : 39, "", mobile ? 14 : 16, orange,
    mobile ? { "font-weight": 600 } : { "text-anchor": "end", "font-weight": 600 });
  label(g, mobile ? 17 : 24, mobile ? 81 : 66, "Measurements on AIME", mobile ? 13 : 16, muted);
  label(g, mobile ? 17 : 24, mobile ? 103 : 88, "Higher and further left is better", mobile ? 13 : 16, muted);
  const xy = ([error, success]) => [x0 + error / 85 * (x1 - x0), y1 - (success - 12) / 10 * (y1 - y0)];
  for (const tick of [0, 20, 40, 60, 80]) {
    const x = x0 + tick / 85 * (x1 - x0);
    line(g, x, y0, x, y1, "#e9e1d8");
    label(g, x, y1 + (mobile ? 20 : 25), String(tick), mobile ? 12 : 15, muted, { "text-anchor": "middle" });
  }
  for (const tick of [12, 14, 16, 18, 20, 22]) {
    const y = y1 - (tick - 12) / 10 * (y1 - y0);
    line(g, x0, y, x1, y, "#e9e1d8");
    label(g, x0 - 12, y + 5, String(tick), mobile ? 12 : 15, muted, { "text-anchor": "end" });
  }
  line(g, x0, y0, x0, y1, "#9f8a76", 2);
  line(g, x0, y1, x1, y1, "#9f8a76", 2);
  label(g, x1, y1 + (mobile ? 42 : 55), "Error@1 (%) →", mobile ? 14 : 16, ink, { "text-anchor": "end" });
  label(g, mobile ? 17 : 24, mobile ? 127 : 113, "Success@1 ↑", mobile ? 13 : 16, ink);
  const historic = [200, 400].map((stepValue, i) => {
    const hGroup = group(g);
    const points = cautions.map((c) => xy(frontier[stepValue][c]));
    hGroup.append(node("polyline", { points: points.map(formatPoint).join(" "), fill: "none", stroke: i ? orange : blue,
      "stroke-width": mobile ? 2 : 3, "stroke-opacity": 0.37, "stroke-linejoin": "round" }));
    for (const point of points) circle(hGroup, ...point, mobile ? 2.5 : 3, i ? orange : blue);
    return hGroup;
  });
  const currentLine = node("polyline", { fill: "none", stroke: blue, "stroke-width": mobile ? 2.8 : 4, "stroke-linejoin": "round", "stroke-linecap": "round" });
  g.append(currentLine);
  const dots = cautions.map(() => circle(g, 0, 0, mobile ? 5 : 7, blue, "#fff", mobile ? 1.5 : 2));
  const cLabels = [0, 0.8].map((c) => label(g, 0, 0, `c = ${c}`, mobile ? 12 : 16, ink, { "text-anchor": "middle", "font-weight": 600 }));
  return { g, compute, historic, currentLine, dots, cLabels, xy };
}

function cotCard(parent, x, y, w, h, { title, topic, lines, outcome, color }, mobile) {
  const card = group(parent, `translate(${x} ${y})`);
  label(card, mobile ? 11 : 16, mobile ? 21 : 29, title, mobile ? 16 : 21, color, { "font-weight": 600, class: "storyboard-display" });
  label(card, mobile ? 12 : 17, mobile ? 46 : 68, topic.toUpperCase(), mobile ? 11.5 : 16, muted, { "font-weight": 600 });
  const start = mobile ? 61 : 91;
  const lineHeight = mobile ? 13 : 23;
  lines.forEach((value, index) => label(card, mobile ? 12 : 17, start + index * lineHeight, value, mobile ? 12 : 20, ink));
  const outcomeY = mobile ? h - 25 : 178;
  label(card, mobile ? 12 : 17, outcomeY + (mobile ? 16 : 21), outcome, mobile ? 15 : 21, color, { "font-weight": 600 });
}

function decisionScene(parent, mobile) {
  const w = mobile ? 340 : 535;
  const g = group(parent);
  label(g, mobile ? 15 : 20, mobile ? 29 : 39, "Explicit Verbalization in the CoT", mobile ? 19 : 23, ink, { "font-weight": 600, class: "storyboard-display" });
  const left = {
    title: "RLVR incentivizes guessing", topic: "Square partition", color: blue,
    lines: [
      "This seems complicated. Maybe P is at",
      "(½, ½), the center, but that gives 8",
      "triangles, not 10. … Given the time",
      "constraints,",
    ],
    outcome: "I'll guess 1.",
  };
  const right = {
    title: "GracefulRL acknowledges limitations", topic: "Hexagon", color: orange,
    lines: [
      "I might need more advanced geometric",
      "techniques or specific theorems that",
      "I'm not currently recalling. I'm not",
      "confident in solving this problem.",
    ],
    outcome: "I don't know.",
  };
  cotCard(g, mobile ? 12 : 18, mobile ? 43 : 62, mobile ? 316 : 499, mobile ? 133 : 215, left, mobile);
  line(g, mobile ? 12 : 18, mobile ? 180 : 282, mobile ? w - 12 : w - 18, mobile ? 180 : 282, rule, 1);
  cotCard(g, mobile ? 12 : 18, mobile ? 186 : 289, mobile ? 316 : 499, mobile ? 133 : 215, right, mobile);
  return g;
}

function makeBoard(mobile) {
  const root = group(svg);
  const simplex = group(root);
  let panels;
  if (mobile) {
    panels = methods.map((method, index) => simplexPanel(group(simplex, `translate(20 ${index ? 325 : 10}) scale(0.7727)`), method));
    mobileLegend(simplex);
  } else {
    const scene = group(simplex, "translate(60 20) scale(1.2)");
    panels = methods.map((method, index) => simplexPanel(group(scene, `translate(${index * 460} 0)`), method));
    desktopLegend(scene);
  }
  const chart = group(root, mobile ? "translate(20 180)" : "translate(320 60)");
  const plot = frontierScene(chart, mobile);
  const decisions = group(root, mobile ? "translate(20 465)" : "translate(625 75)");
  decisionScene(decisions, mobile);
  chart.setAttribute("opacity", "0");
  decisions.setAttribute("opacity", "0");
  return { root, simplex, panels, chart, plot, decisions, mobile };
}

const desktop = makeBoard(false);
const phone = makeBoard(true);
const requestedFrame = new URLSearchParams(location.search).get("storyboardFrame");
const previewFrame = requestedFrame !== null && /^\d+$/.test(requestedFrame)
  ? Math.min(99, Number(requestedFrame)) : null;

function renderPanel(panel, step) {
  panel.trails.forEach(({ points, trail, current }) => {
    trail.setAttribute("points", points.slice(0, step + 1).map(formatPoint).join(" "));
    const [x, y] = points[step];
    current.setAttribute("cx", x);
    current.setAttribute("cy", y);
  });
  const average = panel.histories.reduce((sum, history) => sum + history[step][0] +
    simplexData.configuration.caution * history[step][1], 0) / panel.histories.length;
  panel.reward.textContent = average.toFixed(2);
}

function renderFrontier(plot, step) {
  const current = frontierAt(step);
  const color = step < 300 ? blue : step < 600 ? orange : green;
  const points = cautions.map((c) => plot.xy(current[c]));
  plot.currentLine.setAttribute("points", points.map(formatPoint).join(" "));
  plot.currentLine.setAttribute("stroke", color);
  plot.dots.forEach((dot, index) => {
    dot.setAttribute("cx", points[index][0]);
    dot.setAttribute("cy", points[index][1]);
    dot.setAttribute("fill", color);
  });
  plot.historic.forEach((historic, index) => historic.setAttribute("opacity", step > (index ? 400 : 200) ? 1 : 0));
  [0, 0.8].forEach((c, index) => {
    const [x, y] = plot.xy(current[c]);
    plot.cLabels[index].setAttribute("x", x);
    plot.cLabels[index].setAttribute("y", y - (plot === desktop.plot ? 18 : 12));
  });
  plot.compute.textContent = step < 300 ? "200 steps · 1× compute" : step < 600 ? "400 steps · 2× compute" : "800 steps · 4× compute";
}

function renderBoard(board, frame) {
  const f = Math.max(0, Math.min(99, frame));
  const step = f <= 27 ? Math.round(100 * f / 27) : 100;
  board.panels.forEach((p) => renderPanel(p, step));
  let frontierStep = 200;
  if (f > 49 && f <= 58) frontierStep = mix(200, 400, ease((f - 49) / 9));
  else if (f > 58 && f <= 68) frontierStep = mix(400, 800, ease((f - 58) / 10));
  else if (f > 68) frontierStep = 800;
  renderFrontier(board.plot, frontierStep);

  const transition = ease((f - 28) / 7);
  const frontierEntrance = ease((f - 35) / 7);
  board.simplex.setAttribute("opacity", f <= 27 ? 1 : 1 - transition);
  board.chart.setAttribute("opacity", f < 35 ? 0 : frontierEntrance);
  const finalTransition = ease((f - 69) / 13);
  board.decisions.setAttribute("opacity", f < 69 ? 0 : finalTransition);
  if (board.mobile) {
    board.chart.setAttribute("transform", `translate(20 ${mix(180, 10, finalTransition)})`);
    board.decisions.setAttribute("transform", `translate(20 ${mix(495, 465, finalTransition)})`);
  } else {
    board.chart.setAttribute("transform", `translate(${mix(320, 40, finalTransition)} ${mix(60, 75, finalTransition)})`);
    board.decisions.setAttribute("transform", `translate(625 ${mix(99, 75, finalTransition)})`);
  }
}

function setLayout() {
  const mobile = narrow.matches;
  svg.setAttribute("viewBox", mobile ? "0 0 380 800" : "0 0 1200 675");
  svg.setAttribute("aria-description", mobile ? "Vertically arranged animated simplex, reliability frontier, and model decisions." : "Animated simplex, reliability frontier, and model decisions.");
  desktop.root.style.display = mobile ? "none" : "";
  phone.root.style.display = mobile ? "" : "none";
}
setLayout();
narrow.addEventListener("change", () => { setLayout(); renderCurrent(); });

let currentFrame = previewFrame ?? (motion.matches ? 99 : 0);
let elapsedInCycle = Math.min(currentFrame, firstHoldFrame) * simplexFrameDuration +
  Math.max(0, Math.min(currentFrame, finalFrame) - firstHoldFrame) * laterFrameDuration +
  (currentFrame > firstHoldFrame ? firstHoldDuration : 0);
let pausedByUser = previewFrame !== null || motion.matches;
let visible = false;
let animation = null;
let startTime = null;
function renderCurrent() {
  renderBoard(narrow.matches ? phone : desktop, currentFrame);
}
function updateControl() {
  const playing = animation !== null;
  playPause.textContent = playing ? "Pause" : "Play";
  playPause.setAttribute("aria-label", playing ? "Pause animation" : "Play animation");
}
function stop() {
  if (animation !== null) cancelAnimationFrame(animation);
  animation = null;
  startTime = null;
  updateControl();
}
function tick(time) {
  if (startTime === null) startTime = time - elapsedInCycle;
  elapsedInCycle = (time - startTime) % cycleDuration;
  if (elapsedInCycle < firstHoldStart) currentFrame = elapsedInCycle / simplexFrameDuration;
  else if (elapsedInCycle < firstHoldStart + firstHoldDuration) currentFrame = firstHoldFrame;
  else currentFrame = Math.min(finalFrame, firstHoldFrame +
    (elapsedInCycle - firstHoldStart - firstHoldDuration) / laterFrameDuration);
  renderCurrent();
  animation = requestAnimationFrame(tick);
}
function start() {
  if (animation !== null || pausedByUser || !visible || document.hidden) return;
  animation = requestAnimationFrame(tick);
  updateControl();
}
renderBoard(desktop, currentFrame);
renderBoard(phone, currentFrame);
controls.hidden = false;
updateControl();
playPause.addEventListener("click", () => {
  if (animation !== null) {
    pausedByUser = true;
    stop();
  } else {
    pausedByUser = false;
    start();
  }
});
motion.addEventListener("change", () => {
  if (motion.matches) {
    pausedByUser = true;
    stop();
    currentFrame = 99;
    elapsedInCycle = animationDuration + firstHoldDuration;
    renderCurrent();
  } else {
    pausedByUser = false;
    currentFrame = 0;
    elapsedInCycle = 0;
    renderCurrent();
    start();
  }
});
document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); else start(); });
new IntersectionObserver(([entry]) => {
  visible = entry.isIntersecting;
  if (visible) start(); else stop();
}, { threshold: 0.05 }).observe(svg);
