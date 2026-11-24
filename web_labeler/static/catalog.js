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
  const viewMiro = document.getElementById("catalogViewMiro");
  const viewer = document.getElementById("catalogViewer");
  const viewNoImage = document.getElementById("catalogViewNoImage");
  const viewEvidence = document.getElementById("catalogViewEvidence");
  const personInput = document.getElementById("catalogPersonInput");
  const pasteBox = document.getElementById("catalogPaste");
  const pasteImg = document.getElementById("catalogPasteImg");
  const pasteName = document.getElementById("catalogPasteName");
  const pasteMsg = document.getElementById("catalogPasteMsg");
  const pasteSave = document.getElementById("catalogPasteSave");
  const toast = document.getElementById("catalogToast");
  const PERSON_KEY = "catalog.person";
  const EVIDENCE_FILTERS = [["pending", "Čeka potvrdu"], ["confirmed", "Potvrđeno"], ["rejected", "Odbijeno"], ["all", "Sve"]];
  let evidenceFilter = "pending";
  let selectedVariation = null;
  let pastePayload = null;
  const MIRO_FILTERS = [["all", "Sve"], ["waiting", "Čeka potvrdu"], ["confirmed", "Potvrđeno"], ["none", "Bez klase"]];
  let miroFilter = "waiting";
  let exampleSeed = Math.floor(Math.random() * 1e9);
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

  function imageCell(v) {
    const td = el("td");
    if (v.image) {
      const img = el("img", "catalogThumb");
      img.src = v.image.url;
      img.alt = v.image.source_label;
      img.title = v.image.source_label;
      img.onclick = (e) => { e.stopPropagation(); openViewer(v.image.url, `${v.image.source_label}: ${v.name}`); };
      td.append(img);
    } else if (v.pending_images) {
      td.append(el("span", "catalogNoImage", "čeka potvrdu"));
    } else {
      td.append(el("span", "catalogNoImage", "nema slike"));
    }
    return td;
  }

  function selectHint() {
    const hint = el("div", "catalogPasteHint", "Klikni na artikal u tabeli, pa pritisni Ctrl+V da dodaš sliku sa snimka.");
    hint.id = "catalogPasteHint";
    return hint;
  }

  function updateHint() {
    const hint = document.getElementById("catalogPasteHint");
    if (!hint) return;
    hint.textContent = selectedVariation
      ? `Izabrano: ${selectedVariation.name} (šifra ${selectedVariation.code || "nema"}). Pritisni Ctrl+V da dodaš sliku.`
      : "Klikni na artikal u tabeli, pa pritisni Ctrl+V da dodaš sliku sa snimka.";
    hint.classList.toggle("active", !!selectedVariation);
  }

  function variationsTable(items, withState, context) {
    const table = el("table", context ? "catalogTable selectable" : "catalogTable");
    const head = el("tr");
    for (const title of ["Artikal", "Šifra", "Prodato", "Poslednja prodaja", withState ? "Stanje" : "Oznake", "Slika"]) {
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
        el("td", "num", v.code ? formatQty(v.sold) : "nema"), el("td", "", formatDate(v.last_sale)), tags, imageCell(v));
      if (context) {
        if (selectedVariation && selectedVariation.identity_id === v.identity_id) row.classList.add("selected");
        row.onclick = () => {
          table.querySelectorAll("tr.selected").forEach((x) => x.classList.remove("selected"));
          row.classList.add("selected");
          selectedVariation = { ...v, ...context(v) };
          updateHint();
        };
      }
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
    const parts = [title, facts, renderMiroBlock(data), renderExamplesBlock(data), renderRuleSections(data)];
    if (roleSelect.value === "admin" && data.findings.length) {
      const box = el("div", "catalogRule");
      box.append(el("h3", "", "Neslaganja izvora"));
      data.findings.forEach((f) => box.append(findingCard(f)));
      parts.push(box);
    }
    const heading = el("h3", "catalogVarHead", `POS varijacije (${data.variations.length})`);
    parts.push(heading);
    if (data.united) parts.push(el("div", "catalogSubhead", "Spisak važi za celu spojenu klasu, pa ne znači da je svaki artikal baš ova varijanta."));
    if (data.variations.length) {
      parts.push(selectHint(), variationsTable(data.variations, false, () => ({ model: data.model, cls: data.class })));
    } else {
      parts.push(el("div", "catalogHint", "Nijedan POS artikal nije vezan za ovu klasu."));
    }
    target.replaceChildren(...parts);
  }

  function openViewer(src, caption) {
    viewer.querySelector("img").src = src;
    viewer.querySelector("figcaption").textContent = caption;
    viewer.classList.remove("hidden");
  }
  viewer.onclick = () => { viewer.classList.add("hidden"); viewer.querySelector("img").removeAttribute("src"); };

  function sourceBadge(text) { return el("span", "catalogBadge", text); }

  function pageCaption(p) { return `${p.model}, strana ${p.page_no}`; }

  function miroThumb(p, big) {
    const fig = el("figure", big ? "catalogPage big" : "catalogPage");
    const img = el("img");
    img.loading = "lazy";
    img.src = p.image_url;
    img.alt = `${p.source_label}: ${pageCaption(p)}`;
    img.onclick = () => openViewer(p.image_url, `${p.source_label}, ${pageCaption(p)}`);
    const cap = el("figcaption");
    cap.append(sourceBadge(p.source_label), document.createTextNode(` ${pageCaption(p)}`));
    if (!p.confirmed) cap.append(el("span", "catalogTag warn", "predlog, admin još nije potvrdio"));
    fig.append(img, cap);
    return fig;
  }

  function renderMiroBlock(data) {
    const box = el("div", "catalogImages");
    box.append(el("h3", "", "Stara Miro strana"));
    if (!data.miro_pages.length) {
      box.append(el("div", "catalogHint small", "Nijedna stara Miro strana nije vezana za ovu klasu."));
      return box;
    }
    const row = el("div", "catalogPageRow");
    data.miro_pages.forEach((p) => row.append(miroThumb(p, true)));
    box.append(row);
    return box;
  }

  function renderExamplesBlock(data) {
    const box = el("div", "catalogImages");
    const head = el("h3", "", "Primeri iz dataseta ");
    const again = el("button", "btn small", "Drugi primeri");
    head.append(again);
    const note = el("div", "catalogSubhead", "Nasumični isečci iz dataseta. Pokazuju klasu, ne određen POS artikal.");
    const row = el("div", "catalogExampleRow");
    box.append(head, note, row);
    const load = async () => {
      row.replaceChildren(el("div", "catalogHint small", "Učitavam…"));
      try {
        const url = `/api/catalog/models/${encodeURIComponent(data.model)}/classes/${encodeURIComponent(data.class)}/examples?count=6&seed=${exampleSeed}`;
        const ex = await getJson(url);
        if (!ex.available) { row.replaceChildren(el("div", "catalogHint small", "Dataset nije dostupan na ovom računaru.")); again.hidden = true; return; }
        if (!ex.examples.length) { row.replaceChildren(el("div", "catalogHint small", "Nema primera u datasetu za ovu klasu.")); again.hidden = true; return; }
        row.replaceChildren(...ex.examples.map((e) => {
          const fig = el("figure", "catalogExample");
          const img = el("img");
          img.loading = "lazy";
          img.src = e.url;
          img.alt = e.source_label;
          img.onclick = () => openViewer(e.url, `${e.source_label}, klasa ${data.class}`);
          const cap = el("figcaption");
          cap.append(sourceBadge(e.source_label));
          fig.append(img, cap);
          return fig;
        }));
      } catch (e) {
        row.replaceChildren(el("div", "catalogHint small", e.message));
      }
    };
    again.onclick = () => { exampleSeed = Math.floor(Math.random() * 1e9); load(); };
    load();
    return box;
  }

  async function postJson(url, payload) {
    let response;
    try {
      response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    } catch (_) { throw new Error("Server trenutno nije dostupan. Pokušaj ponovo."); }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "Nije sačuvano.");
    return data;
  }

  function miroMatches(p) {
    if (miroFilter === "waiting") return !p.confirmed;
    if (miroFilter === "confirmed") return p.confirmed;
    if (miroFilter === "none") return !p.classes.length;
    return true;
  }

  function miroCard(p, classes, onSaved) {
    const card = el("div", p.confirmed ? "catalogMiroCard done" : "catalogMiroCard");
    card.append(miroThumb(p, false));
    const side = el("div", "catalogMiroSide");
    side.append(el("div", "catalogMiroStatus", p.confirmed ? "Potvrđeno" : "Čeka potvrdu"));
    if (p.heading) side.append(el("div", "catalogMiroHeading", `Na strani piše: ${p.heading}`));
    if (p.low_confidence) side.append(el("span", "catalogTag warn", "predlog nije siguran"));
    if (p.note) side.append(el("div", "catalogSubhead", p.note));
    if (!p.confirmed) {
      side.append(el("div", "catalogSubhead", p.suggested.length ? `Predlog: ${p.suggested.join(", ")}` : "Nema predloga."));
    }
    const chosen = new Set(p.classes);
    const chips = el("div", "catalogChips");
    const drawChips = () => {
      chips.replaceChildren();
      if (!chosen.size) chips.append(el("span", "catalogHint small", "Bez klase"));
      for (const name of [...chosen].sort()) {
        const chip = el("span", "catalogChip", name);
        const x = el("button", "", "×");
        x.title = "Ukloni";
        x.onclick = () => { chosen.delete(name); drawChips(); };
        chip.append(x);
        chips.append(chip);
      }
    };
    drawChips();
    const add = el("select", "catalogAdd");
    add.append(new Option("Dodaj klasu…", ""), ...(classes[p.model] || []).map((n) => new Option(n, n)));
    add.onchange = () => { if (add.value) chosen.add(add.value); add.value = ""; drawChips(); };
    const buttons = el("div", "catalogMiroButtons");
    const save = el("button", "btn primary", "Potvrdi");
    const msg = el("span", "catalogSubhead");
    save.onclick = async () => {
      save.disabled = true;
      try { onSaved(await postJson(`/api/catalog/miro-pages/${encodeURIComponent(p.key)}`, { classes: [...chosen] })); }
      catch (e) { msg.textContent = e.message; save.disabled = false; }
    };
    buttons.append(save);
    if (p.confirmed) {
      const undo = el("button", "btn", "Vrati na predlog");
      undo.onclick = async () => {
        undo.disabled = true;
        try { onSaved(await postJson(`/api/catalog/miro-pages/${encodeURIComponent(p.key)}`, { reset: true })); }
        catch (e) { msg.textContent = e.message; undo.disabled = false; }
      };
      buttons.append(undo);
    }
    buttons.append(msg);
    side.append(chips, add, buttons);
    card.append(side);
    return card;
  }

  function renderMiro(data) {
    const model = modelSelect.value;
    const title = el("h2", "catalogTitle", "Stare Miro strane ");
    title.append(el("span", "catalogCount",
      `potvrđeno ${data.counts.confirmed} od ${data.counts.total}, bez klase ${data.counts.without_class}`));
    const intro = el("div", "catalogSubhead",
      "Za svaku stranu potvrdi kojoj klasi pripada. Strana može da ide uz više klasa, a možeš je potvrditi i bez klase. Svaka potvrda se odmah čuva.");
    const filters = el("div", "catalogViews");
    for (const [key, label] of MIRO_FILTERS) {
      const b = el("button", key === miroFilter ? "btn primary small" : "btn small", label);
      b.onclick = () => { miroFilter = key; renderMiro(data); };
      filters.append(b);
    }
    const grid = el("div", "catalogMiroGrid");
    const pages = data.pages.filter((p) => p.model === model && miroMatches(p));
    for (const p of pages) {
      grid.append(miroCard(p, data.classes, (saved) => {
        const i = data.pages.findIndex((x) => x.key === saved.key);
        data.pages[i] = saved;
        data.counts.confirmed = data.pages.filter((x) => x.confirmed).length;
        data.counts.without_class = data.pages.filter((x) => !x.classes.length).length;
        renderMiro(data);
      }));
    }
    if (!pages.length) grid.append(el("div", "catalogHint", "Nema strana za ovaj izbor."));
    body.replaceChildren(title, intro, filters, grid);
  }

  function readPerson() {
    try { return localStorage.getItem(PERSON_KEY) || ""; } catch (_) { return ""; }
  }
  function savePerson(name) {
    personInput.value = name;
    try { localStorage.setItem(PERSON_KEY, name); } catch (_) { /* private mode */ }
  }
  personInput.value = readPerson();
  personInput.onchange = () => savePerson(personInput.value.trim());

  function showToast(text) {
    toast.textContent = text;
    toast.classList.remove("hidden");
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => toast.classList.add("hidden"), 4000);
  }

  function closePaste() {
    pasteBox.classList.add("hidden");
    pasteImg.removeAttribute("src");
    pastePayload = null;
  }
  document.getElementById("catalogPasteCancel").onclick = closePaste;

  function clipboardImage(e) {
    const items = [...(e.clipboardData ? e.clipboardData.items : [])];
    const item = items.find((i) => i.kind === "file" && i.type.startsWith("image/"));
    return item ? item.getAsFile() : null;
  }

  document.addEventListener("paste", (e) => {
    if (panel.classList.contains("hidden") || !pasteBox.classList.contains("hidden")) return;
    if (e.target.closest && e.target.closest("input, textarea, select")) return;
    if (view !== "classes" && view !== "noimage") return;
    e.preventDefault();
    if (!selectedVariation) { showToast("Prvo klikni na artikal u tabeli, pa onda Ctrl+V."); return; }
    const file = clipboardImage(e);
    if (!file) { showToast("U klipbordu nema slike. Napravi screenshot dela ekrana, pa pritisni Ctrl+V."); return; }
    if (file.size > 10 * 1024 * 1024) { showToast("Slika je prevelika (najviše 10 MB). Iseci manji deo ekrana."); return; }
    const reader = new FileReader();
    reader.onload = () => {
      const v = selectedVariation;
      pastePayload = { identity_id: v.identity_id, image_base64: reader.result, model: v.model || null, class: v.cls || null };
      pasteImg.src = reader.result;
      document.getElementById("catalogPasteWhat").textContent =
        `${v.name}, šifra ${v.code || "nema"}${v.cls ? `, klasa ${v.cls}` : ""}`;
      pasteName.value = personInput.value.trim();
      pasteMsg.textContent = "";
      pasteSave.disabled = false;
      pasteBox.classList.remove("hidden");
      (pasteName.value ? pasteSave : pasteName).focus();
    };
    reader.readAsDataURL(file);
  });

  pasteSave.onclick = async () => {
    const name = pasteName.value.trim();
    if (!name) { pasteMsg.textContent = "Upiši svoje ime."; pasteName.focus(); return; }
    savePerson(name);
    pasteSave.disabled = true;
    try {
      await postJson("/api/catalog/evidence", { ...pastePayload, added_by: name, role: roleSelect.value });
      closePaste();
      showToast("Slika je sačuvana i čeka potvrdu admina.");
      loadClasses();
    } catch (e) {
      pasteMsg.textContent = e.message;
      pasteSave.disabled = false;
    }
  };

  function formatTime(value) {
    if (!value) return "";
    const d = new Date(value);
    return d.toLocaleString("sr-RS", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  const ACTION_LABELS = { added: "dodato", confirmed: "potvrđeno", rejected: "odbijeno" };

  function evidenceCard(item, onSaved) {
    const card = el("div", `catalogMiroCard ${item.status === "pending" ? "" : item.status === "confirmed" ? "done" : "rejected"}`);
    const fig = el("figure", "catalogPage");
    const img = el("img");
    img.src = item.url;
    img.alt = item.source_label;
    img.onclick = () => openViewer(item.url, `${item.source_label}: ${item.name}`);
    const cap = el("figcaption");
    cap.append(sourceBadge(item.source_label));
    fig.append(img, cap);
    const side = el("div", "catalogMiroSide");
    side.append(el("div", "catalogMiroStatus", item.status_label),
      el("div", "catalogMiroHeading", `${item.name}, šifra ${item.code || "nema"}`));
    if (item.class) side.append(el("div", "catalogSubhead", `Dodato na klasi ${item.model} / ${item.class}`));
    const hist = el("ul", "catalogHistory");
    for (const h of item.history) {
      const li = el("li", "", `${formatTime(h.at)}, ${ACTION_LABELS[h.action] || h.action}: ${h.actor}`);
      if (h.note) li.append(el("span", "catalogSubhead", ` (${h.note})`));
      hist.append(li);
    }
    side.append(hist);
    if (item.status === "pending") {
      const reason = el("input", "catalogReason");
      reason.placeholder = "Razlog odbijanja (nije obavezno)";
      const msg = el("span", "catalogSubhead");
      const buttons = el("div", "catalogMiroButtons");
      const decide = async (decision, button) => {
        const name = personInput.value.trim();
        if (!name) { msg.textContent = "Upiši svoje ime gore desno."; personInput.focus(); return; }
        button.disabled = true;
        try {
          onSaved(await postJson(`/api/catalog/evidence/${item.evidence_id}/decision`,
            { decision, decided_by: name, reason: reason.value }));
        } catch (e) { msg.textContent = e.message; button.disabled = false; }
      };
      const ok = el("button", "btn primary", "Potvrdi");
      ok.onclick = () => decide("confirmed", ok);
      const no = el("button", "btn", "Odbij");
      no.onclick = () => decide("rejected", no);
      buttons.append(ok, no, msg);
      side.append(reason, buttons);
    }
    card.append(fig, side);
    return card;
  }

  function renderEvidence(data) {
    const title = el("h2", "catalogTitle", "Nove slike ");
    title.append(el("span", "catalogCount",
      `čeka ${data.counts.pending}, potvrđeno ${data.counts.confirmed}, odbijeno ${data.counts.rejected}`));
    const intro = el("div", "catalogSubhead",
      "Potvrđena slika postaje dokaz za taj artikal i skida oznaku „nije provereno na snimku”. Odbijena slika ostaje u istoriji.");
    const filters = el("div", "catalogViews");
    for (const [key, label] of EVIDENCE_FILTERS) {
      const b = el("button", key === evidenceFilter ? "btn primary small" : "btn small", label);
      b.onclick = () => { evidenceFilter = key; loadClasses(); };
      filters.append(b);
    }
    const grid = el("div", "catalogMiroGrid");
    for (const item of data.items) grid.append(evidenceCard(item, () => loadClasses()));
    if (!data.items.length) grid.append(el("div", "catalogHint", "Nema slika za ovaj izbor."));
    body.replaceChildren(title, intro, filters, grid);
  }

  function renderNoImage(model, data) {
    const title = el("h2", "catalogTitle", `Bez slike: ${model} `);
    title.append(el("span", "catalogCount", `${data.items.length} artikala, najprodavaniji na vrhu`));
    const intro = el("div", "catalogSubhead", "Praćeni artikli ovog modela koji još nemaju potvrđenu sliku.");
    if (!data.items.length) {
      body.replaceChildren(title, el("div", "catalogHint", "Svi praćeni artikli ovog modela imaju potvrđenu sliku."));
      return;
    }
    const firstClass = (v) => {
      const own = v.classes.find((c) => c.split(".")[0] === model.toLowerCase());
      return { model, cls: own ? own.split(".").slice(1).join(".") : null };
    };
    const table = variationsTable(data.items, false, firstClass);
    const head = table.querySelector("thead tr");
    head.insertBefore(el("th", "", "Klasa"), head.children[1]);
    table.querySelectorAll("tbody tr").forEach((row, i) => {
      row.insertBefore(el("td", "", data.items[i].classes.join(", ")), row.children[1]);
    });
    body.replaceChildren(title, intro, selectHint(), table);
    updateHint();
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
      if (view === "evidence") {
        const data = await getJson(`/api/catalog/evidence?status=${evidenceFilter}`);
        if (no === requestNo) renderEvidence(data);
        return;
      }
      if (view === "noimage") {
        const model = modelSelect.value;
        const data = await getJson(`/api/catalog/without-image?model=${encodeURIComponent(model)}`);
        if (no === requestNo) renderNoImage(model, data);
        return;
      }
      if (view === "miro") {
        const data = await getJson("/api/catalog/miro-pages");
        if (no === requestNo) renderMiro(data);
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
    viewMiro.classList.toggle("primary", view === "miro");
    viewNoImage.classList.toggle("primary", view === "noimage");
    viewEvidence.classList.toggle("primary", view === "evidence");
    selectedVariation = null;
    modelSelect.closest("label").classList.toggle("hidden", !["classes", "miro", "noimage"].includes(view));
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
    if (!open) { viewer.classList.add("hidden"); closePaste(); }
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
    viewMiro.classList.toggle("hidden", !admin);
    viewEvidence.classList.toggle("hidden", !admin);
    if (!admin && ["report", "miro", "evidence"].includes(view)) setView("classes");
    else if (!panel.classList.contains("hidden")) loadClasses();
  }
  roleSelect.onchange = () => {
    try { localStorage.setItem(ROLE_KEY, roleSelect.value); } catch (_) { /* private mode */ }
    applyRole();
  };
  viewReport.classList.toggle("hidden", roleSelect.value !== "admin");
  viewMiro.classList.toggle("hidden", roleSelect.value !== "admin");
  viewMiro.onclick = () => setView("miro");
  viewEvidence.classList.toggle("hidden", roleSelect.value !== "admin");
  viewEvidence.onclick = () => setView("evidence");
  viewNoImage.onclick = () => setView("noimage");
  viewReport.onclick = () => setView("report");
  modelSelect.onchange = () => { selectedClass = null; selectedVariation = null; loadClasses(); };
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
