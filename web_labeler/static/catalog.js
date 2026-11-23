// Classes tab. A separate layer over the workspace: it never calls setMode or touches the loaded
// video and annotations, so opening it cannot lose unexported work.
(function () {
  const ROLE_KEY = "catalog.role";
  const tab = document.getElementById("tabClasses");
  const tabs = tab.closest(".tabs");
  const panel = document.getElementById("catalogPanel");
  const modelSelect = document.getElementById("catalogModelSelect");
  const roleSelect = document.getElementById("catalogRoleSelect");
  const body = document.getElementById("catalogBody");
  const viewClasses = document.getElementById("catalogViewClasses");
  const viewUnclassified = document.getElementById("catalogViewUnclassified");
  const viewReport = document.getElementById("catalogViewReport");
  let requestNo = 0;
  let view = "classes";
  let selectedClass = null;

  function readRole() {
    try { return localStorage.getItem(ROLE_KEY) === "admin" ? "admin" : "labeler"; }
    catch (_) { return "labeler"; }
  }

  function showMessage(text) {
    const box = document.createElement("div");
    box.className = "catalogMessage";
    box.textContent = text;
    body.replaceChildren(box);
  }

  async function getJson(url) {
    let response;
    try { response = await fetch(url); }
    catch (_) { throw new Error("Server trenutno nije dostupan. Pokušaj ponovo."); }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "Katalog trenutno ne može da se pročita.");
    return data;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function formatDate(value) {
    if (!value) return "nema";
    const [y, m, d] = value.slice(0, 10).split("-");
    return `${d}.${m}.${y}.`;
  }

  function formatQty(value) {
    return Number(value || 0).toLocaleString("sr-RS", { maximumFractionDigits: 2 });
  }

  function variationsTable(items, withState) {
    const table = el("table", "catalogTable");
    const head = el("tr");
    for (const title of ["Artikal", "Šifra", "Prodato", "Poslednja prodaja", withState ? "Stanje" : "Oznake"]) {
      const th = el("th", title === "Prodato" ? "num" : "", title);
      head.append(th);
    }
    const thead = el("thead");
    thead.append(head);
    const tbody = el("tbody");
    for (const v of items) {
      const row = el("tr");
      const tags = el("td");
      if (withState) tags.append(el("span", "catalogTag", v.state_label));
      for (const tag of v.tags) {
        const chip = el("span", tag === "privremeno" ? "catalogTag warn" : "catalogTag", tag);
        if (tag === "privremeno" && v.provisional_reason) chip.title = v.provisional_reason;
        tags.append(chip);
      }
      row.append(el("td", "", v.name), el("td", "", v.code || "nema"),
        el("td", "num", v.code ? formatQty(v.sold) : "nema"), el("td", "", formatDate(v.last_sale)), tags);
      tbody.append(row);
    }
    table.append(thead, tbody);
    return table;
  }

  function renderRuleSections(detail) {
    const box = el("div", "catalogRules");
    if (!detail.has_description) {
      box.append(el("div", "catalogNoDesc", "Ova klasa još nema opis kako se prepoznaje."));
      return box;
    }
    for (const section of detail.rules) {
      const part = el("div", "catalogRule");
      part.append(el("h3", "", section.label));
      for (const item of section.items) {
        const row = el("p", "", item.text);
        const origin = item.from_class ? `${item.source}, klasa ${item.from_class}` : item.source;
        row.append(el("span", "catalogSource", origin));
        part.append(row);
      }
      box.append(part);
    }
    return box;
  }

  function findingCard(f) {
    const card = el("div", `catalogFinding ${f.severity}`);
    card.append(el("div", "catalogFindingTitle", f.title), el("p", "", f.text));
    const list = el("ul");
    for (const source of f.sources) {
      const item = el("li");
      item.append(el("strong", "", `${source.label}: `), document.createTextNode(source.says));
      list.append(item);
    }
    card.append(list);
    return card;
  }

  function renderDetail(data, target) {
    const title = el("h2", "catalogTitle", data.class);
    const facts = el("div", "catalogFacts");
    facts.append(el("span", data.training_state === "trained" ? "catalogTag" : "catalogTag warn", data.training_label));
    if (data.trained_name && data.trained_name !== data.class) {
      facts.append(el("span", "catalogTag", `U modelu: ${data.trained_name}`));
    }
    if (data.united) facts.append(el("span", "catalogTag", `Labeleri crtaju posebno: ${data.drawn_as.join(", ")}`));
    const parts = [title, facts, renderRuleSections(data)];
    if (roleSelect.value === "admin" && data.findings.length) {
      const box = el("div", "catalogRule");
      box.append(el("h3", "", "Neslaganja izvora"));
      data.findings.forEach((f) => box.append(findingCard(f)));
      parts.push(box);
    }
    const heading = el("h3", "catalogVarHead", `POS varijacije (${data.variations.length})`);
    parts.push(heading);
    if (data.united) parts.push(el("div", "catalogSubhead", "Spisak važi za celu spojenu klasu, pa ne znači da je svaki artikal baš ova varijanta."));
    parts.push(data.variations.length ? variationsTable(data.variations, false)
      : el("div", "catalogHint", "Nijedan POS artikal nije vezan za ovu klasu."));
    target.replaceChildren(...parts);
  }

  async function showVariations(model, className, detail, button) {
    const no = ++requestNo;
    body.querySelectorAll(".catalogList .selected").forEach((b) => b.classList.remove("selected"));
    button.classList.add("selected");
    selectedClass = className;
    detail.replaceChildren(el("div", "catalogHint", "Učitavam…"));
    try {
      const url = `/api/catalog/models/${encodeURIComponent(model)}/classes/${encodeURIComponent(className)}/details`;
      const data = await getJson(url);
      if (no === requestNo) renderDetail(data, detail);
    } catch (e) {
      if (no === requestNo) detail.replaceChildren(el("div", "catalogHint", e.message));
    }
  }

  function classButton(model, c, detail) {
    const button = el("button", "catalogClass");
    if (c.id !== undefined) button.append(el("span", "catalogClassId", c.id));
    const label = el("span", "catalogClassName", c.name);
    if (c.training_state === "excluded" || c.training_state === "not_in_model") {
      label.append(el("small", "warn", c.training_label));
    } else if (c.trained_name && c.trained_name !== c.name) {
      label.append(el("small", "", `u modelu ${c.trained_name}`));
    }
    button.append(label, el("span", "catalogVarCount", c.variation_count));
    button.onclick = () => showVariations(model, c.name, detail, button);
    return button;
  }

  function renderClasses(model, data) {
    const list = el("div", "catalogList");
    const detail = el("div", "catalogDetail");
    const title = el("h2", "catalogTitle", `${model} `);
    title.append(el("span", "catalogCount", `${data.classes.length} klasa`));
    list.append(title);
    const buttons = new Map();
    for (const c of data.classes) {
      const button = classButton(model, c, detail);
      buttons.set(c.name, button);
      list.append(button);
    }
    if (data.map_only_classes.length) {
      list.append(el("div", "catalogSubhead", "Klase iz POS mape koje još nisu povezane sa listom"));
      for (const c of data.map_only_classes) {
        const button = classButton(model, c, detail);
        buttons.set(c.name, button);
        list.append(button);
      }
    }
    const split = el("div", "catalogSplit");
    split.append(list, detail);
    body.replaceChildren(split);
    const again = buttons.get(selectedClass);
    if (again) showVariations(model, selectedClass, detail, again);
    else detail.append(el("div", "catalogHint", "Izaberi klasu da vidiš njene POS varijacije."));
  }

  function renderReport(data) {
    const title = el("h2", "catalogTitle", "Razlike izvora ");
    title.append(el("span", "catalogCount",
      `${data.counts.high} važno, ${data.counts.medium} za proveru, ${data.counts.info} informacija`));
    const nodes = [title];
    for (const [severity, label] of [["high", "Važno"], ["medium", "Za proveru"], ["info", "Informacije"]]) {
      const items = data.findings.filter((f) => f.severity === severity);
      if (!items.length) continue;
      const section = el("details", "catalogGroup");
      section.open = severity === "high";
      section.append(el("summary", "", `${label} (${items.length})`));
      for (const f of items) {
        const card = findingCard(f);
        if (f.model) card.prepend(el("div", "catalogFindingModel", f.model));
        section.append(card);
      }
      nodes.push(section);
    }
    body.replaceChildren(...nodes);
  }

  function renderUnclassified(data) {
    const nodes = [];
    for (const group of data.groups) {
      const section = el("details", "catalogGroup");
      section.open = group.state === "pending";
      section.append(el("summary", "", `${group.label} (${group.items.length})`));
      section.append(group.items.length ? variationsTable(group.items, true) : el("div", "catalogHint", "Nema artikala."));
      nodes.push(section);
    }
    body.replaceChildren(...nodes);
  }

  async function loadClasses() {
    const no = ++requestNo;
    try {
      if (view === "unclassified") {
        const data = await getJson("/api/catalog/unclassified");
        if (no === requestNo) renderUnclassified(data);
        return;
      }
      if (view === "report") {
        const data = await getJson("/api/catalog/source-report");
        if (no === requestNo) renderReport(data);
        return;
      }
      const model = modelSelect.value;
      if (!model) return;
      const data = await getJson(`/api/catalog/models/${encodeURIComponent(model)}/classes`);
      if (no === requestNo) renderClasses(model, data);
    } catch (e) {
      if (no === requestNo) showMessage(e.message);
    }
  }

  function setView(next) {
    view = next;
    viewClasses.classList.toggle("primary", view === "classes");
    viewUnclassified.classList.toggle("primary", view === "unclassified");
    viewReport.classList.toggle("primary", view === "report");
    modelSelect.closest("label").classList.toggle("hidden", view !== "classes");
    loadClasses();
  }

  async function refresh() {
    const no = ++requestNo;
    const previous = modelSelect.value;
    modelSelect.disabled = true;
    try {
      const status = await getJson("/api/catalog/status");
      if (no !== requestNo) return;
      if (!status.available) { modelSelect.replaceChildren(); showMessage(status.message); return; }
      const data = await getJson("/api/catalog/models");
      if (no !== requestNo) return;
      modelSelect.replaceChildren(...data.models.map((m) => new Option(m.name, m.name)));
      if (data.models.some((m) => m.name === previous)) modelSelect.value = previous;
      modelSelect.disabled = false;
      await loadClasses();
    } catch (e) {
      if (no === requestNo) { modelSelect.replaceChildren(); showMessage(e.message); }
    }
  }

  function setOpen(open) {
    panel.classList.toggle("hidden", !open);
    tabs.classList.toggle("catalogOpen", open);
    tab.classList.toggle("tabActive", open);
  }

  // A running server without the Catalog routes still serves this page: keep the tab hidden there.
  fetch("/api/catalog/status").then((r) => { tab.hidden = !r.ok; }).catch(() => {});

  roleSelect.value = readRole();
  function applyRole() {
    const admin = roleSelect.value === "admin";
    viewReport.classList.toggle("hidden", !admin);
    if (!admin && view === "report") setView("classes");
    else if (!panel.classList.contains("hidden")) loadClasses();
  }
  roleSelect.onchange = () => {
    try { localStorage.setItem(ROLE_KEY, roleSelect.value); } catch (_) { /* private mode */ }
    applyRole();
  };
  viewReport.classList.toggle("hidden", roleSelect.value !== "admin");
  viewReport.onclick = () => setView("report");
  modelSelect.onchange = () => { selectedClass = null; loadClasses(); };
  viewClasses.onclick = () => setView("classes");
  viewUnclassified.onclick = () => setView("unclassified");
  tab.onclick = () => { setOpen(true); refresh(); };
  tabs.addEventListener("click", (e) => {
    const button = e.target.closest(".tabBtn");
    if (button && button !== tab) setOpen(false);
  }, true);
  // The labeling shortcuts listen on window: keep them from acting on the hidden workspace.
  for (const type of ["keydown", "keyup"]) {
    window.addEventListener(type, (e) => {
      if (!panel.classList.contains("hidden")) e.stopPropagation();
    }, true);
  }
})();
