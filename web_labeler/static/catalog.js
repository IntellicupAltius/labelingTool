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
  const viewArticles = document.getElementById("catalogViewArticles");
  const viewDecisions = document.getElementById("catalogViewDecisions");
  const viewModelCards = document.getElementById("catalogViewModelCards");
  const viewReleases = document.getElementById("catalogViewReleases");
  const viewFindings = document.getElementById("catalogViewFindings");
  const viewDays = document.getElementById("catalogViewDays");
  const viewInbox = document.getElementById("catalogViewInbox");
  const tabInbox = document.getElementById("tabInbox");
  const FINDING_FILTERS = [["open", "Čeka odluku"], ["all", "Sve"]];
  let findingFilter = "open";
  const NOT_RECORDED = "nije zabeleženo";
  let openCardFile = null;
  let cardRetries = 0;
  const ARTICLE_FILTERS = [["", "Svi"], ["tracked", "Praćeno"], ["missing", "Mapiranje nedostaje"], ["pending", "Čeka odluku"],
    ["dormant", "Uspavano"], ["ignored", "Ignorisano"], ["new_class", "Nova klasa"]];
  const DECISION_FILTERS = [["propose", "Predlozi"], ["ignore", "X odluke"], ["", "Sve odluke"]];
  let articleQuery = "";
  let articleFilter = "";
  let decisionFilter = "propose";
  let openArticle = null;
  let classChoices = null;
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
      const nameCell = el("td");
      const link = el("button", "catalogLink", v.name);
      link.title = "Otvori karticu artikla";
      link.onclick = (e) => { e.stopPropagation(); showArticle(v.article_id); };
      nameCell.append(link);
      row.append(nameCell, el("td", "", v.code || "nema"),
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
    if (!response.ok) {
      const detail = data.detail && typeof data.detail === "object" ? data.detail.message : data.detail;
      throw new Error(detail || "Nije sačuvano.");
    }
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

  function isAdmin() { return roleSelect.value === "admin"; }

  function person() {
    const name = personInput.value.trim();
    if (!name) { personInput.focus(); throw new Error("Upiši svoje ime gore desno."); }
    return name;
  }

  async function decide(body) {
    return postJson("/api/catalog/decisions", { ...body, actor: person(), role: roleSelect.value });
  }

  async function loadChoices() {
    if (!classChoices) classChoices = (await getJson("/api/catalog/class-choices")).models;
    return classChoices;
  }

  function stateTag(v) {
    return el("span", v.state === "tracked" ? "catalogTag" : "catalogTag warn", v.state_label);
  }

  function mappingCell(v) {
    const td = el("td", v.state !== "ignored" && !v.classes.length ? "catalogMissing" : "", v.mapping);
    return td;
  }

  function renderArticles(data) {
    const title = el("h2", "catalogTitle", "Artikli ");
    title.append(el("span", "catalogCount", `${data.items.length} prikazano, mapiranje nedostaje kod ${data.counts.missing}`));
    const search = el("input", "catalogSearch");
    search.type = "search";
    search.placeholder = "Traži po imenu ili šifri";
    search.value = articleQuery;
    let timer = null;
    search.oninput = () => { clearTimeout(timer); timer = setTimeout(() => { articleQuery = search.value; loadClasses(true); }, 250); };
    const filters = el("div", "catalogViews");
    for (const [key, label] of ARTICLE_FILTERS) {
      const count = key === "" ? "" : ` (${data.counts[key]})`;
      const b = el("button", key === articleFilter ? "btn primary small" : "btn small", label + count);
      b.onclick = () => { articleFilter = key; loadClasses(); };
      filters.append(b);
    }
    const table = el("table", "catalogTable");
    const head = el("tr");
    for (const t of ["Artikal", "Šifra", "Prodato", "Poslednja prodaja", "Stanje", "Mapiranje", "Oznake"]) head.append(el("th", t === "Prodato" ? "num" : "", t));
    const thead = el("thead");
    thead.append(head);
    const tbody = el("tbody");
    for (const v of data.items) {
      const row = el("tr", "catalogClickRow");
      const tags = el("td");
      v.tags.forEach((t) => tags.append(el("span", t === "privremeno" ? "catalogTag warn" : "catalogTag", t)));
      row.append(el("td", "", v.name), el("td", "", v.code || "nema"), el("td", "num", v.code ? formatQty(v.sold) : "nema"),
        el("td", "", formatDate(v.last_sale)), el("td"), mappingCell(v), tags);
      row.children[4].append(stateTag(v));
      row.onclick = () => showArticle(v.article_id);
      tbody.append(row);
    }
    table.append(thead, tbody);
    const intro = el("div", "catalogSubhead", "Svi POS artikli lokala. Klik na red otvara karticu artikla.");
    const nodes = [title, intro, search, filters, data.items.length ? table : el("div", "catalogHint", "Nema artikala za ovaj izbor.")];
    body.replaceChildren(...nodes);
    search.focus();
    search.setSelectionRange(search.value.length, search.value.length);
  }

  function historyItem(h, card) {
    const item = el("div", `catalogDecision${h.revoked ? " revoked" : ""}`);
    const head = el("div", "catalogDecisionHead");
    head.append(el("strong", "", h.kind_label), document.createTextNode(` ${h.subject}${h.code ? `, šifra ${h.code}` : ""}: ${h.text}`));
    item.append(head);
    const meta = `${formatTime(h.created_at)}, ${h.actor} (${h.role === "admin" ? "admin" : "labeler"}), važi od ${formatDate(h.valid_from)}`;
    item.append(el("div", "catalogSubhead", meta));
    if (h.reason) item.append(el("div", "", `Zašto: ${h.reason}`));
    if (h.status_label) item.append(el("span", h.status === "pending" ? "catalogTag warn" : "catalogTag", h.status_label));
    if (h.revoked) item.append(el("span", "catalogTag warn", `poništeno (odluka #${h.revoked_by})`));
    if (h.future) item.append(el("span", "catalogTag", "još ne važi"));
    if (isAdmin() && h.miro_text) {
      const box = el("div", "catalogMiroText");
      box.append(el("div", "catalogSubhead", "Tekst za Miro i Word (do zaključavanja):"), el("div", "", h.miro_text));
      const copy = el("button", "btn small", "Kopiraj");
      copy.onclick = () => navigator.clipboard.writeText(h.miro_text).then(() => showToast("Tekst je kopiran."), () => showToast("Kopiranje nije uspelo, označi tekst ručno."));
      box.append(copy);
      item.append(box);
    }
    if (isAdmin()) {
      const buttons = el("div", "catalogMiroButtons");
      const msg = el("span", "catalogSubhead");
      if (["classes", "provisional", "ignore", "merge", "class_target"].includes(h.kind) && !h.revoked) {
        const undo = el("button", "btn small", "Poništi");
        undo.onclick = () => askReason(item, "Zašto poništavaš?", async (reason) => {
          await decide({ kind: "revoke", target_id: h.decision_id, reason });
        });
        buttons.append(undo);
      }
      if (h.kind === "propose" && h.status === "pending") {
        const change = h.payload.change || {};
        const applyBtn = (label, star) => {
          const b = el("button", star ? "btn small" : "btn primary small", label);
          b.onclick = () => askReason(item, star ? "Zašto primenjuješ i šta nije sigurno?" : "Zašto primenjuješ?", async (reason) => {
            const payload = { ...change, from_proposal: h.decision_id };
            if (star) payload.provisional = reason;
            await decide({ kind: h.payload.kind, identity_id: h.identity_id, article_id: h.article_id, payload, reason });
          });
          return b;
        };
        if (h.payload.kind === "new_class") {
          const openBtn = el("button", "btn primary small", "Otvori klasu");
          openBtn.onclick = () => classTargetForm(item, { identity_id: h.identity_id, article_id: h.article_id, proposal_id: h.decision_id });
          buttons.append(openBtn);
        } else buttons.append(applyBtn("Primeni", false));
        if (h.payload.kind === "classes") buttons.append(applyBtn("Primeni sa zvezdicom", true));
        const no = el("button", "btn small", "Odbij");
        no.onclick = () => askReason(item, "Zašto odbijaš?", async (reason) => {
          await decide({ kind: "reject", target_id: h.decision_id, reason });
        });
        buttons.append(no);
      }
      if (card === undefined && h.article_id) {
        const open = el("button", "btn small", "Otvori karticu");
        open.onclick = () => showArticle(h.article_id);
        buttons.append(open);
      }
      if (buttons.children.length) { buttons.append(msg); item.append(buttons); }
    } else if (card === undefined && h.article_id) {
      const open = el("button", "btn small", "Otvori karticu");
      open.onclick = () => showArticle(h.article_id);
      item.append(open);
    }
    return item;
  }

  function askReason(container, label, action) {
    container.querySelectorAll(".catalogAsk").forEach((x) => x.remove());
    const box = el("div", "catalogAsk");
    const input = el("input", "catalogReason");
    input.placeholder = label;
    const ok = el("button", "btn primary small", "Sačuvaj");
    const cancel = el("button", "btn small", "Odustani");
    const msg = el("span", "catalogPasteMsg");
    ok.onclick = async () => {
      ok.disabled = true;
      try { await action(input.value.trim()); showToast("Sačuvano."); refreshAfterDecision(); }
      catch (e) { msg.textContent = e.message; ok.disabled = false; }
    };
    cancel.onclick = () => box.remove();
    box.append(input, ok, cancel, msg);
    container.append(box);
    input.focus();
  }

  function refreshAfterDecision() {
    if (openArticle !== null) showArticle(openArticle);
    else loadClasses();
  }

  function linkEditor(model, start) {
    const box = el("div", "catalogLinkEditor");
    const chosen = [...start];
    const chips = el("div", "catalogChips");
    const draw = () => {
      chips.replaceChildren();
      if (!chosen.length) chips.append(el("span", "catalogHint small", "Bez klase (mapiranje nedostaje)"));
      chosen.forEach((c, i) => {
        const chip = el("span", "catalogChip", `${i === 0 ? "glavna" : "rezerva"}: ${c.model}.${c.class}`);
        const x = el("button", "", "×");
        x.onclick = () => { chosen.splice(i, 1); draw(); };
        chip.append(x);
        chips.append(chip);
      });
    };
    const modelPick = el("select");
    const classPick = el("select");
    const fillClasses = () => {
      classPick.replaceChildren(new Option("Izaberi klasu…", ""), ...(classChoices[modelPick.value] || []).map((n) => new Option(n, n)));
    };
    modelPick.append(...Object.keys(classChoices).map((m) => new Option(m, m)));
    modelPick.value = model && classChoices[model] ? model : Object.keys(classChoices)[0];
    modelPick.onchange = fillClasses;
    fillClasses();
    classPick.onchange = () => {
      if (classPick.value && !chosen.some((c) => c.model === modelPick.value && c.class === classPick.value)) {
        chosen.push({ model: modelPick.value, class: classPick.value });
      }
      classPick.value = "";
      draw();
    };
    draw();
    const pickers = el("div", "catalogMiroButtons");
    pickers.append(el("span", "catalogSubhead", "Dodaj klasu u modelu:"), modelPick, classPick);
    box.append(chips, pickers);
    return { box, links: () => chosen };
  }

  function variationForm(container, v, proposal) {
    container.querySelectorAll(".catalogForm").forEach((x) => x.remove());
    const form = el("div", "catalogForm");
    form.append(el("h4", "", proposal ? `Predlog promene za ${v.name}` : `Promena klase za ${v.name}`));
    const start = v.classes.map((c) => { const [m, ...rest] = c.split("."); return { model: m, class: rest.join(".") }; });
    const editor = linkEditor(start.length ? start[0].model : null, start);
    form.append(editor.box);
    const mode = el("select");
    mode.append(new Option("Prvo glavna, pa rezerva (fallback)", "fallback"), new Option("Bilo koja od navedenih (any)", "any"));
    mode.value = v.mapping_mode === "any" ? "any" : "fallback";
    const star = el("input", "catalogReason");
    star.placeholder = "Zvezdica: šta nije sigurno (prazno = bez zvezdice)";
    star.value = v.provisional_reason || "";
    const from = el("input");
    from.type = "date";
    from.value = new Date().toISOString().slice(0, 10);
    const reason = el("input", "catalogReason");
    reason.placeholder = "Zašto (obavezno)";
    const fields = el("div", "catalogFormGrid");
    fields.append(el("label", "", "Ako ima više klasa"), mode, el("label", "", "Zvezdica"), star);
    if (!proposal) fields.append(el("label", "", "Važi od"), from);
    fields.append(el("label", "", "Zašto"), reason);
    const msg = el("span", "catalogPasteMsg");
    const save = el("button", "btn primary", proposal ? "Pošalji predlog" : "Sačuvaj odluku");
    const cancel = el("button", "btn", "Odustani");
    cancel.onclick = () => form.remove();
    save.onclick = async () => {
      save.disabled = true;
      const change = { links: editor.links(), mode: mode.value, provisional: star.value.trim() || null };
      try {
        if (proposal) await decide({ kind: "propose", identity_id: v.identity_id, payload: { kind: "classes", change }, reason: reason.value });
        else await decide({ kind: "classes", identity_id: v.identity_id, payload: change, reason: reason.value, valid_from: from.value });
        showToast(proposal ? "Predlog je poslat adminu." : "Odluka je sačuvana. Čeka sledeće izdanje.");
        refreshAfterDecision();
      } catch (e) { msg.textContent = e.message; save.disabled = false; }
    };
    const buttons = el("div", "catalogMiroButtons");
    buttons.append(save, cancel, msg);
    form.append(fields, buttons);
    container.append(form);
  }

  function mergeForm(container, card) {
    container.querySelectorAll(".catalogForm").forEach((x) => x.remove());
    const form = el("div", "catalogForm");
    form.append(el("h4", "", `Spoji ${card.name} sa postojećim artiklom`),
      el("div", "catalogSubhead", "Koristi kad je isto piće dobilo novu šifru ili novo ime u kasi."));
    const search = el("input", "catalogSearch");
    search.placeholder = "Traži artikal po imenu ili šifri";
    const results = el("div", "catalogResults");
    let target = null;
    const reason = el("input", "catalogReason");
    reason.placeholder = "Zašto (obavezno)";
    const msg = el("span", "catalogPasteMsg");
    let timer = null;
    search.oninput = () => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        if (search.value.trim().length < 2) { results.replaceChildren(); return; }
        const data = await getJson(`/api/catalog/articles?q=${encodeURIComponent(search.value.trim())}`);
        const seen = new Set();
        results.replaceChildren(...data.items.filter((v) => v.article_id !== card.home_article_id && !seen.has(v.article_id) && seen.add(v.article_id))
          .slice(0, 12).map((v) => {
            const b = el("button", "catalogResult", `${v.name} (šifra ${v.code || "nema"}, ${v.mapping})`);
            b.onclick = () => { target = v; results.querySelectorAll(".selected").forEach((x) => x.classList.remove("selected")); b.classList.add("selected"); };
            return b;
          }));
      }, 250);
    };
    const save = el("button", "btn primary", "Spoji");
    save.onclick = async () => {
      if (!target) { msg.textContent = "Izaberi artikal sa spiska."; return; }
      save.disabled = true;
      try {
        await decide({ kind: "merge", article_id: card.home_article_id, payload: { into: target.article_id }, reason: reason.value });
        showToast("Artikli su spojeni. Čeka sledeće izdanje.");
        showArticle(target.article_id);
      } catch (e) { msg.textContent = e.message; save.disabled = false; }
    };
    const cancel = el("button", "btn", "Odustani");
    cancel.onclick = () => form.remove();
    const buttons = el("div", "catalogMiroButtons");
    buttons.append(save, cancel, msg);
    form.append(search, results, reason, buttons);
    container.append(form);
  }

  function renderArticle(card) {
    const back = el("button", "btn small", "← Nazad");
    back.onclick = () => { openArticle = null; loadClasses(); };
    const title = el("h2", "catalogTitle", card.home_name);
    const nodes = [back, title];
    if (card.merged_into) nodes.push(el("div", "catalogSubhead", `Artikal ${card.name} je spojen sa ovim artiklom.`));
    const actions = el("div", "catalogMiroButtons");
    const formSlot = el("div");
    const x = el("button", "btn", "X Ne pratimo");
    x.title = "Za hranu, cigarete i sve što kamera ne treba da prati";
    x.onclick = () => askReason(formSlot, "Napomena (nije obavezno), npr. hrana", async (reason) => {
      await decide({ kind: "ignore", article_id: card.home_article_id, reason });
    });
    actions.append(x);
    const gone = el("button", "btn", "Izbačeno iz asortimana");
    gone.title = "Bilo je u ponudi, više se ne prodaje. Ostaje u Katalogu sa istorijom, ne prati se.";
    gone.onclick = () => askReason(formSlot, "Napomena (nije obavezno), npr. skinuto sa menija u aprilu", async (reason) => {
      await decide({ kind: "ignore", article_id: card.home_article_id, reason, payload: { why: "assortment" } });
    });
    actions.append(gone);
    if (isAdmin()) {
      const merge = el("button", "btn", "Spoji sa…");
      merge.onclick = () => mergeForm(formSlot, card);
      actions.append(merge);
    }
    nodes.push(actions, formSlot);
    nodes.push(el("h3", "catalogVarHead", `POS varijacije (${card.variations.length})`));
    const table = el("table", "catalogTable");
    const head = el("tr");
    for (const t of ["Ime u kasi", "Šifra", "Prodato", "Prvi put", "Poslednji put", "Stanje", "Mapiranje", "Oznake", "Slika", ""]) head.append(el("th", t === "Prodato" ? "num" : "", t));
    const thead = el("thead");
    thead.append(head);
    const tbody = el("tbody");
    for (const v of card.variations) {
      const row = el("tr");
      const tags = el("td");
      v.tags.forEach((t) => {
        const chip = el("span", t === "privremeno" ? "catalogTag warn" : "catalogTag", t);
        if (t === "privremeno" && v.provisional_reason) chip.title = v.provisional_reason;
        tags.append(chip);
      });
      const mapping = mappingCell(v);
      if (v.mode_label) mapping.append(el("div", "catalogSubhead", v.mode_label));
      const act = el("td");
      const change = el("button", "btn small", isAdmin() ? "Promeni klasu" : "Predloži promenu");
      change.onclick = async () => { await loadChoices(); variationForm(formSlot, v, !isAdmin()); formSlot.scrollIntoView({ block: "nearest" }); };
      act.append(change);
      if (isAdmin() && v.state !== "ignored") {
        const fresh = el("button", "btn small", "Nova klasa…");
        fresh.title = "Otvori novu klasu za ovo piće ili izaberi klasu koju model još uči";
        fresh.onclick = () => { classTargetForm(formSlot, { identity_id: v.identity_id, article_id: v.article_id }); formSlot.scrollIntoView({ block: "nearest" }); };
        act.append(fresh);
      }
      if (v.has_clips) {
        const watch = el("button", "btn small", "Snimak");
        watch.title = "Snimak prodaje sa 3 kamere; odavde i slanje na označavanje";
        watch.onclick = () => { clipViewer(formSlot, v.identity_id); formSlot.scrollIntoView({ block: "nearest" }); };
        act.append(watch);
      }
      row.append(el("td", "", v.name), el("td", "", v.code || "nema"), el("td", "num", v.code ? formatQty(v.sold) : "nema"),
        el("td", "", formatDate(v.first_seen)), el("td", "", formatDate(v.last_seen)), el("td"), mapping, tags, imageCell(v), act);
      row.children[5].append(stateTag(v));
      tbody.append(row);
    }
    table.append(thead, tbody);
    nodes.push(table);
    if (card.images.length) {
      nodes.push(el("h3", "catalogVarHead", "Slike"));
      const grid = el("div", "catalogExampleRow");
      card.images.forEach((img) => {
        const fig = el("figure", "catalogExample");
        const pic = el("img");
        pic.src = img.url;
        pic.onclick = () => openViewer(img.url, `${img.source_label}: ${img.name}`);
        const cap = el("figcaption");
        cap.append(sourceBadge(img.source_label), document.createTextNode(` ${img.status_label}`));
        fig.append(pic, cap);
        grid.append(fig);
      });
      nodes.push(grid);
    }
    nodes.push(el("h3", "catalogVarHead", `Istorija odluka (${card.history.length})`));
    if (!card.history.length) nodes.push(el("div", "catalogHint small", "Još nema ručnih odluka. Stanje dolazi iz POS izveštaja i mape."));
    card.history.forEach((h) => nodes.push(historyItem(h, card)));
    body.replaceChildren(...nodes);
  }

  async function showArticle(articleId) {
    const no = ++requestNo;
    openArticle = articleId;
    body.replaceChildren(el("div", "catalogHint", "Učitavam…"));
    try {
      const card = await getJson(`/api/catalog/articles/${articleId}`);
      if (no === requestNo) renderArticle(card);
    } catch (e) {
      if (no === requestNo) showMessage(e.message);
    }
  }

  function renderDecisions(data) {
    const title = el("h2", "catalogTitle", "Odluke ");
    const pending = data.items.filter((d) => d.kind === "propose" && d.status === "pending").length;
    title.append(el("span", "catalogCount", decisionFilter === "propose" ? `${pending} predloga čeka odluku` : `${data.items.length} zapisa`));
    const intro = el("div", "catalogSubhead", "Ništa se ne briše. Promena ili poništavanje pravi novi zapis, a odluke čekaju sledeće izdanje i ne menjaju produkciju.");
    const filters = el("div", "catalogViews");
    for (const [key, label] of DECISION_FILTERS) {
      const b = el("button", key === decisionFilter ? "btn primary small" : "btn small", label);
      b.onclick = () => { decisionFilter = key; loadClasses(); };
      filters.append(b);
    }
    const list = el("div", "catalogDecisionList");
    data.items.forEach((h) => list.append(historyItem(h)));
    if (!data.items.length) list.append(el("div", "catalogHint", "Nema zapisa."));
    body.replaceChildren(title, intro, filters, list);
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

  // ---------------- New class (admin) ----------------

  async function classTargetForm(container, target) {
    container.querySelectorAll(".catalogAsk").forEach((x) => x.remove());
    const box = el("div", "catalogAsk catalogClassForm");
    const msg = el("span", "catalogPasteMsg");
    const modelPick = el("select");
    const kindPick = el("select");
    kindPick.append(new Option("Nova klasa", "new"), new Option("Postojeća klasa", "existing"));
    const nameInput = el("input", "catalogReason");
    nameInput.placeholder = "Ime nove klase, npr. MATCHA_CASA";
    const classPick = el("select");
    const reason = el("input", "catalogReason");
    reason.placeholder = "Zašto (npr. novo piće na meniju)";
    const hint = el("div", "catalogSubhead",
      "Nova klasa ide na kraj liste klasa u alatu, brojevi postojećih klasa se ne menjaju. Artikal se uparuje tek kad promovisan model nauči klasu.");
    try {
      const models = (await getJson("/api/catalog/models")).models;
      modelPick.append(new Option("Izaberi model…", ""), ...models.map((m) => new Option(m.name, m.name)));
    } catch (e) { msg.textContent = e.message; }
    const fillClasses = async () => {
      classPick.replaceChildren(new Option("Izaberi klasu…", ""));
      if (!modelPick.value) return;
      const view = await getJson(`/api/catalog/models/${encodeURIComponent(modelPick.value)}/classes`);
      for (const c of view.classes) {
        if (c.training_state === "excluded") continue;
        classPick.append(new Option(`${c.name} (${c.training_label})`, c.name));
      }
    };
    const showKind = () => {
      const fresh = kindPick.value === "new";
      nameInput.style.display = fresh ? "" : "none";
      classPick.style.display = fresh ? "none" : "";
    };
    modelPick.onchange = () => fillClasses().catch((e) => { msg.textContent = e.message; });
    kindPick.onchange = showKind;
    showKind();
    const save = el("button", "btn primary small", "Sačuvaj");
    const cancel = el("button", "btn small", "Odustani");
    save.onclick = async () => {
      save.disabled = true;
      try {
        const fresh = kindPick.value === "new";
        const cls = fresh ? nameInput.value.trim() : classPick.value;
        if (!modelPick.value) throw new Error("Izaberi model.");
        if (!cls) throw new Error(fresh ? "Upiši ime nove klase." : "Izaberi klasu.");
        const payload = { model: modelPick.value, class: cls, open_new: fresh };
        if (target.proposal_id) payload.from_proposal = target.proposal_id;
        await decide({ kind: "class_target", identity_id: target.identity_id, article_id: target.article_id, payload, reason: reason.value.trim() });
        showToast(fresh ? "Klasa je otvorena." : "Sačuvano.");
        refreshAfterDecision();
      } catch (e) { msg.textContent = e.message; save.disabled = false; }
    };
    cancel.onclick = () => box.remove();
    box.append(modelPick, kindPick, nameInput, classPick, reason, save, cancel, msg, hint);
    container.append(box);
  }

  // ---------------- Inbox "čeka te" ----------------

  function inboxForm(slot, v, kind) {
    slot.replaceChildren();
    const box = el("div", "catalogAsk");
    const msg = el("span", "catalogPasteMsg");
    const note = el("input", "catalogReason");
    note.placeholder = "Napomena (nije obavezno)";
    let read = () => ({});
    if (kind === "classes") {
      const modelPick = el("select");
      const classPick = el("select");
      const fill = () => classPick.replaceChildren(new Option("Izaberi klasu…", ""), ...(classChoices[modelPick.value] || []).map((n) => new Option(n, n)));
      modelPick.append(...Object.keys(classChoices).map((m) => new Option(m, m)));
      modelPick.onchange = fill;
      fill();
      box.append(modelPick, classPick);
      read = () => {
        if (!classPick.value) throw new Error("Izaberi klasu.");
        return { links: [{ model: modelPick.value, class: classPick.value }] };
      };
    } else if (kind === "merge") {
      const pick = el("input", "catalogReason");
      pick.placeholder = "Ime postojećeg artikla";
      const list = el("datalist");
      list.id = `catalogMerge${v.identity_id}`;
      pick.setAttribute("list", list.id);
      getJson("/api/catalog/articles?state=tracked").then((d) => {
        const seen = new Set();
        for (const a of d.items) if (a.article_id !== v.article_id && !seen.has(a.name)) { seen.add(a.name); list.append(new Option(a.name, a.name)); }
        pick._items = d.items;
      });
      box.append(pick, list);
      read = () => {
        const hit = (pick._items || []).find((a) => a.name === pick.value.trim());
        if (!hit) throw new Error("Izaberi artikal sa spiska.");
        return { into: hit.article_id };
      };
    }
    const save = el("button", "btn primary small", "Pošalji");
    const cancel = el("button", "btn small", "Odustani");
    save.onclick = async () => {
      save.disabled = true;
      try {
        const change = read();
        if (kind === "new_class") change.note = note.value.trim();
        await decide({ kind: "propose", identity_id: v.identity_id, article_id: v.article_id, payload: { kind, change }, reason: note.value.trim() || "predlog labelera" });
        showToast("Poslato adminu.");
        loadClasses();
      } catch (e) { msg.textContent = e.message; save.disabled = false; }
    };
    cancel.onclick = () => slot.replaceChildren();
    box.append(note, save, cancel, msg);
    slot.append(box);
  }

  async function renderInbox(data) {
    await loadChoices();
    const title = el("h2", "catalogTitle", "Čeka te ");
    title.append(el("span", "catalogCount", `${data.counts.waiting} artikala`));
    const intro = el("div", "catalogSubhead",
      `Artikli koji se prodaju, a nemaju klasu. Najprodavaniji su prvi. Poslato adminu: ${data.counts.proposed}, čeka novu klasu: ${data.counts.waiting_class}, odloženo: ${data.counts.snoozed}.`);
    const nodes = [title, intro];
    if (data.new_class.length) {
      nodes.push(el("h3", "catalogVarHead", `Nova klasa (${data.new_class.length})`));
      nodes.push(el("div", "catalogSubhead", "Put novog pića: Čeka klasu → U označavanju → Čeka trening → Praćeno."));
      for (const v of data.new_class) {
        const card = el("div", "catalogFinding");
        const head = el("div", "catalogFindingTitle", `${v.name} `);
        head.append(el("span", v.flow.stage === "waiting_class" ? "catalogTag warn" : "catalogTag", v.flow.label));
        card.append(head);
        const where = v.flow.class ? `klasa ${v.flow.model}.${v.flow.class}` : (v.flow.note ? `napomena: ${v.flow.note}` : "klasa još nije izabrana");
        const images = v.flow.raw_images ? ` · označenih slika: ${v.flow.raw_images}` : "";
        card.append(el("div", "catalogSubhead", `Šifra ${v.code || "nema"} · ${where}${images}`));
        const slot = el("div");
        const buttons = el("div", "catalogMiroButtons");
        if (isAdmin() && v.flow.stage === "waiting_class") {
          const openBtn = el("button", "btn primary small", "Otvori klasu");
          openBtn.onclick = () => classTargetForm(slot, { identity_id: v.identity_id, article_id: v.article_id, proposal_id: v.flow.proposal_id });
          buttons.append(openBtn);
        }
        const open = el("button", "btn small", "Otvori karticu");
        open.onclick = () => showArticle(v.article_id);
        buttons.append(open);
        if (v.clip && ["ready", "seen"].includes(v.clip.status)) {
          const watch = el("button", "btn small", "Snimak i slanje na označavanje");
          watch.onclick = () => clipViewer(slot, v.identity_id);
          buttons.append(watch);
        }
        card.append(buttons, slot);
        nodes.push(card);
      }
      nodes.push(el("h3", "catalogVarHead", "Bez klase"));
    }
    if (!data.items.length) nodes.push(el("div", "catalogHint", "Ništa ne čeka. Bravo!"));
    for (const v of data.items) {
      const card = el("div", "catalogFinding");
      const itemHead = el("div", "catalogFindingTitle", `${v.name} `);
      if (v.clip) itemHead.append(el("span", ["missing", "late"].includes(v.clip.status) ? "catalogTag warn" : "catalogTag", v.clip.label));
      card.append(itemHead);
      card.append(el("div", "catalogSubhead", `Šifra ${v.code || "nema"} · prodato ${formatQty(v.sold)} · poslednja prodaja ${formatDate(v.last_sale)}`));
      const slot = el("div");
      const buttons = el("div", "catalogMiroButtons");
      const x = el("button", "btn small", "X Ne pratimo");
      x.title = "Hrana, cigarete i sve što kamera ne prati";
      x.onclick = () => askReason(slot, "Napomena (nije obavezno), npr. hrana", async (reason) => {
        await decide({ kind: "ignore", article_id: v.article_id, reason });
      });
      const propose = el("button", "btn small", "Predloži klasu");
      propose.onclick = () => inboxForm(slot, v, "classes");
      const same = el("button", "btn small", "Isto kao postojeći artikal");
      same.onclick = () => inboxForm(slot, v, "merge");
      const fresh = el("button", "btn small", "Treba nova klasa");
      fresh.onclick = () => inboxForm(slot, v, "new_class");
      const later = el("button", "btn small", "Odloži");
      later.title = "Vraća se pri sledećoj prodaji ili za 14 dana";
      later.onclick = async () => {
        try { await postJson(`/api/catalog/inbox/${v.identity_id}/snooze`, { actor: person(), role: roleSelect.value }); showToast("Odloženo."); loadClasses(); }
        catch (e) { showToast(e.message); }
      };
      const open = el("button", "btn small", "Otvori karticu (slika)");
      open.onclick = () => showArticle(v.article_id);
      buttons.append(x, propose, same, fresh, later, open);
      if (v.clip && ["ready", "seen"].includes(v.clip.status)) {
        const watch = el("button", "btn primary small", "Pogledaj snimak");
        watch.onclick = () => clipViewer(slot, v.identity_id);
        buttons.prepend(watch);
      }
      card.append(buttons, slot);
      nodes.push(card);
    }
    body.replaceChildren(...nodes);
  }

  // ---------------- Footage of a new article: 3 cameras, "Vidim ga", "Nije na snimku" (ticket 17) ----------------

  function fmtSecond(t) {
    const m = Math.floor(t / 60), sec = Math.floor(t % 60);
    return `${m}:${String(sec).padStart(2, "0")}`;
  }

  async function clipViewer(slot, identityId) {
    let card;
    try { card = await getJson(`/api/catalog/clips/${identityId}`); }
    catch (e) { showToast(e.message); return; }
    renderClipCard(slot, card, 0);
  }

  function renderClipCard(slot, card, index) {
    const box = el("div", "catalogClip");
    if (!card.sales.length) {
      box.append(el("div", "catalogHint", card.status_label === "Nema snimka"
        ? "Nema snimka. Program traži sledeću prodaju ovog artikla."
        : `${card.status_label}. Program sprema snimak.`));
      slot.replaceChildren(box);
      return;
    }
    const sale = card.sales[Math.min(index, card.sales.length - 1)];
    const head = el("div", "catalogFindingTitle", `${sale.label} `);
    head.append(el("span", "catalogCount", `prodaja ${index + 1} od ${card.sales.length}`));
    if (sale.seen.length) head.append(el("span", "catalogTag", `viđeno: ${sale.seen.map((s) => s.camera).join(", ")}`));
    const help = el("div", "catalogSubhead", "Snimak kreće od trenutka kucanja. Kad vidiš piće, klikni \"Vidim ga\" i prevuci pravougaonik oko njega na jednoj kameri.");
    const grid = el("div", "catalogClipGrid");
    const videos = [];
    let drawing = false;
    let chosen = null; // {channel, box:[x,y,w,h]}
    const save = el("button", "btn primary small", "Sačuvaj");
    save.disabled = true;
    for (const cam of sale.cameras) {
      const cell = el("div", "catalogClipCell");
      cell.append(el("div", "catalogClipName", cam.name));
      if (!cam.ready) {
        cell.append(el("div", "catalogHint", "Snimak ove kamere se još priprema."));
        grid.append(cell);
        continue;
      }
      const wrap = el("div", "catalogClipWrap");
      const video = document.createElement("video");
      video.src = cam.url; video.muted = true; video.preload = "auto"; video.playsInline = true;
      const canvas = el("canvas", "catalogClipCanvas");
      wrap.append(video, canvas);
      cell.append(wrap);
      grid.append(cell);
      videos.push(video);
      let start = null;
      const point = (ev) => {
        const r = canvas.getBoundingClientRect();
        return [Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height))];
      };
      const paint = (b) => {
        canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!b) return;
        ctx.strokeStyle = "#facc15"; ctx.lineWidth = 3;
        ctx.strokeRect(b[0] * canvas.width, b[1] * canvas.height, b[2] * canvas.width, b[3] * canvas.height);
      };
      canvas.onmousedown = (ev) => {
        if (!drawing) return;
        start = point(ev);
        grid.querySelectorAll("canvas").forEach((c) => { if (c !== canvas) c.getContext("2d").clearRect(0, 0, c.width, c.height); });
      };
      canvas.onmousemove = (ev) => {
        if (!drawing || !start) return;
        const p = point(ev);
        paint([Math.min(start[0], p[0]), Math.min(start[1], p[1]), Math.abs(p[0] - start[0]), Math.abs(p[1] - start[1])]);
      };
      canvas.onmouseup = (ev) => {
        if (!drawing || !start) return;
        const p = point(ev);
        const b = [Math.min(start[0], p[0]), Math.min(start[1], p[1]), Math.abs(p[0] - start[0]), Math.abs(p[1] - start[1])];
        start = null;
        if (b[2] < 0.01 || b[3] < 0.01) { paint(null); return; }
        chosen = { channel: cam.channel, box: b };
        paint(b);
        save.disabled = false;
      };
    }
    // one set of controls for all cameras
    const controls = el("div", "catalogMiroButtons");
    const play = el("button", "btn small", "Pusti");
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = "600"; slider.step = "1"; slider.value = "0"; slider.className = "catalogClipSlider";
    const clock = el("span", "catalogCount", "+0:00 od kucanja");
    const speed = el("button", "btn small", "Brže ×4");
    let rate = 1;
    const lead = () => videos[0];
    const syncAll = (t) => videos.forEach((v) => { if (Math.abs(v.currentTime - t) > 0.4) v.currentTime = t; });
    if (videos.length) {
      lead().addEventListener("loadedmetadata", () => { slider.max = String(Math.floor(lead().duration || 600)); });
      lead().addEventListener("timeupdate", () => {
        slider.value = String(Math.floor(lead().currentTime));
        clock.textContent = `+${fmtSecond(lead().currentTime)} od kucanja`;
        if (!lead().paused) syncAll(lead().currentTime);
      });
    }
    play.onclick = () => {
      if (!videos.length) return;
      if (lead().paused) { syncAll(lead().currentTime); videos.forEach((v) => { v.playbackRate = rate; v.play(); }); play.textContent = "Pauza"; }
      else { videos.forEach((v) => v.pause()); play.textContent = "Pusti"; }
    };
    slider.oninput = () => { const t = Number(slider.value); videos.forEach((v) => { v.currentTime = t; }); clock.textContent = `+${fmtSecond(t)} od kucanja`; };
    speed.onclick = () => { rate = rate === 1 ? 4 : 1; videos.forEach((v) => { v.playbackRate = rate; }); speed.textContent = rate === 1 ? "Brže ×4" : "Normalno"; };
    const seen = el("button", "btn primary small", "Vidim ga");
    const cancel = el("button", "btn small", "Odustani");
    cancel.style.display = "none"; save.style.display = "none";
    seen.onclick = () => {
      videos.forEach((v) => v.pause()); play.textContent = "Pusti";
      drawing = true; chosen = null; save.disabled = true;
      grid.classList.add("drawing");
      seen.style.display = "none"; cancel.style.display = ""; save.style.display = "";
      showToast("Prevuci pravougaonik oko pića na jednoj kameri.");
    };
    cancel.onclick = () => {
      drawing = false; chosen = null; grid.classList.remove("drawing");
      grid.querySelectorAll("canvas").forEach((c) => c.getContext("2d").clearRect(0, 0, c.width, c.height));
      seen.style.display = ""; cancel.style.display = "none"; save.style.display = "none";
    };
    save.onclick = async () => {
      if (!chosen) return;
      try {
        const result = await postJson(`/api/catalog/clips/${card.identity_id}/seen`, {
          sale: sale.slug, channel: chosen.channel, second: lead().currentTime, box: chosen.box, actor: person(), role: roleSelect.value });
        showToast("Sačuvano. Slika čeka potvrdu admina.");
        renderClipCard(slot, result.card, index);
      } catch (e) { showToast(e.message); }
    };
    const notHere = el("button", "btn small", "Nije na snimku");
    notHere.title = "Pića nema na ovom snimku: prelazi se na sledeću prodaju";
    notHere.onclick = async () => {
      try {
        const next = await postJson(`/api/catalog/clips/${card.identity_id}/not-visible`, { sale: sale.slug, actor: person(), role: roleSelect.value });
        renderClipCard(slot, next, 0);
      } catch (e) { showToast(e.message); }
    };
    const nav = el("span");
    if (card.sales.length > 1) {
      const other = el("button", "btn small", "Druga prodaja");
      other.onclick = () => renderClipCard(slot, card, (index + 1) % card.sales.length);
      nav.append(other);
    }
    const close = el("button", "btn small", "Zatvori");
    close.onclick = () => { videos.forEach((v) => v.pause()); slot.replaceChildren(); };
    controls.append(play, slider, clock, speed, seen, cancel, save, notHere, nav, close);
    if (card.target && card.to_send) {
      const send = el("button", "btn primary small", `Pošalji na označavanje (${card.to_send})`);
      send.title = `Isečak 10 sekundi pre i posle svakog "Vidim ga", sa izabranim ${card.target.model} / ${card.target.class}`;
      send.onclick = async () => {
        try {
          send.disabled = true;
          showToast("Pravim isečke, može da potraje minut...");
          const r = await postJson(`/api/catalog/clips/${card.identity_id}/send`, { actor: person(), role: roleSelect.value });
          showToast(`Poslato ${r.sent} isečaka. Otvori ih u Labeler tabu: lista "katalog (za označavanje)".`);
          clipViewer(slot, card.identity_id);
        } catch (e) { send.disabled = false; showToast(e.message); }
      };
      controls.append(send);
    }
    box.append(head, help, grid, controls);
    slot.replaceChildren(box);
  }

  // ---------------- Day table ----------------

  function renderDays(data) {
    const title = el("h2", "catalogTitle", "Dani ");
    title.append(el("span", "catalogCount", data.enabled ? `praćenje od ${formatDate(data.enabled)}` : "praćenje još nije uključeno"));
    const intro = el("div", "catalogSubhead", "Izveštaj za jedan dan treba da stigne do kraja sledećeg dana. Ako lokal nije radio, označi to ovde.");
    const table = el("table", "catalogTable");
    const head = el("tr");
    for (const t of ["Dan", "Stanje", "Redova", "Napomena", ""]) head.append(el("th", "", t));
    table.append(head);
    for (const d of data.days) {
      const tr = el("tr");
      const state = el("td");
      state.append(el("span", ["late", "suspicious"].includes(d.status) ? "catalogTag warn" : "catalogTag", d.label));
      const action = el("td");
      if (d.can_close || d.can_reopen) {
        const b = el("button", "btn small", d.can_reopen ? "Poništi" : "Lokal nije radio");
        b.onclick = () => askReason(tr.lastChild, d.can_reopen ? "Zašto poništavaš?" : "Razlog (npr. praznik)", async (reason) => {
          await postJson(`/api/catalog/day-table/${d.day}`, { action: d.can_reopen ? "reopened" : "closed", actor: person(), role: roleSelect.value, reason });
        });
        action.append(b);
      }
      tr.append(el("td", "", formatDate(d.day)), state, el("td", "num", d.rows === null ? "" : String(d.rows)),
        el("td", "", d.notes.join("; ")), action);
      table.append(tr);
    }
    body.replaceChildren(title, intro, table);
  }

  // ---------------- Findings (POS intake) ----------------

  function renderFindings(data) {
    const title = el("h2", "catalogTitle", "Nalazi ");
    title.append(el("span", "catalogCount", `${data.counts.open} čeka odluku`));
    const intro = el("div", "catalogSubhead", "Sve što je program primetio u POS izveštajima. Ništa se ne menja samo; odluka se može kasnije promeniti.");
    const filters = el("div", "catalogViews");
    for (const [key, label] of FINDING_FILTERS) {
      const b = el("button", key === findingFilter ? "btn primary small" : "btn small", label);
      b.onclick = () => { findingFilter = key; loadClasses(); };
      filters.append(b);
    }
    const nodes = [title, intro, filters];
    if (!data.items.length) nodes.push(el("div", "catalogHint", "Nema nalaza."));
    for (const f of data.items) {
      const card = el("div", `catalogFinding ${f.status === "open" ? "medium" : ""}`);
      card.append(el("div", "catalogFindingTitle", f.title), el("p", "", f.text));
      card.append(el("span", f.status === "open" ? "catalogTag warn" : "catalogTag", f.status_label));
      for (const h of f.history) card.append(el("div", "catalogSubhead", `${formatTime(h.at)}, ${h.actor}: ${({ applied: "primenjeno", denied: "odbijeno", reopened: "vraćeno na odluku" })[h.action]}${h.reason ? ` (${h.reason})` : ""}`));
      const buttons = el("div", "catalogMiroButtons");
      const act = (label, action, primary) => {
        const b = el("button", primary ? "btn primary small" : "btn small", label);
        b.onclick = () => askReason(card, "Napomena (nije obavezno)", async (reason) => {
          await postJson(`/api/catalog/findings/${f.finding_id}`, { action, actor: person(), role: roleSelect.value, reason });
        });
        return b;
      };
      if (f.status === "open") buttons.append(act(f.apply_label, "applied", true), act(f.deny_label, "denied", false));
      else buttons.append(act("Promeni odluku", "reopened", false));
      card.append(buttons);
      nodes.push(card);
    }
    body.replaceChildren(...nodes);
  }

  // ---------------- Releases ----------------

  function checkList(checks) {
    const list = el("div", "catalogChecks");
    for (const c of checks) {
      const row = el("div", c.ok ? "catalogCheck ok" : "catalogCheck bad", `${c.ok ? "✔" : "✖"} ${c.label}`);
      if (!c.ok) {
        const ul = el("ul");
        c.problems.slice(0, 30).forEach((p) => ul.append(el("li", "", p)));
        if (c.problems.length > 30) ul.append(el("li", "", `i još ${c.problems.length - 30}`));
        row.append(ul);
      }
      list.append(row);
    }
    return list;
  }

  function changeList(changes, empty) {
    if (!changes.length) return el("div", "catalogHint small", empty);
    const ul = el("ul", "catalogChangeList");
    changes.forEach((c) => ul.append(el("li", c.kind === "lost" ? "warn" : "", c.text)));
    return ul;
  }

  function parityBlock(parity) {
    const box = el("details", "catalogGroup");
    box.open = parity.unexplained > 0;
    const head = parity.identical ? "Istovetno sa današnjim fajlovima"
      : `Razlike od današnjih fajlova: ${parity.items.length}, neobjašnjeno: ${parity.unexplained}`;
    box.append(el("summary", "", head));
    const counts = Object.entries(parity.counts).map(([area, n]) => n ? null : area).filter(Boolean);
    if (counts.length) box.append(el("div", "catalogSubhead", "Bez ijedne razlike: " + counts.map((a) => ({
      map: "mapa za pipeline", blacklist: "lista za ignorisanje", class_lists: "liste klasa", training: "spajanja i izbacivanja",
      documented: "dokumentovane klase" })[a]).join(", ") + "."));
    for (const item of parity.items) {
      const card = el("div", `catalogFinding ${item.explained ? "" : "high"}`);
      card.append(el("div", "catalogFindingModel", item.area_label));
      card.append(el("div", "catalogFindingTitle", item.text));
      card.append(el("p", "", item.why));
      if (item.details && item.details.length) {
        const ul = el("ul");
        item.details.forEach((d) => ul.append(el("li", "", d)));
        card.append(ul);
      }
      box.append(card);
    }
    return box;
  }

  function releaseRow(r) {
    const card = el("div", `catalogFinding ${r.live ? "medium" : ""}`);
    const kind = r.kind === "rollback" ? `vraćeno na izdanje ${r.based_on}` : "objava";
    card.append(el("div", "catalogFindingTitle", `Izdanje ${r.number}${r.live ? " (važeće)" : ""}`));
    card.append(el("div", "catalogFindingModel", `${formatTime(r.created_at)}, ${r.actor}, ${kind}`));
    card.append(el("p", "", r.reason));
    if (r.intact === false) card.append(el("div", "catalogNoDesc", "Fajlovi na disku se ne slažu sa zapisom izdanja. Javi administratoru."));
    const changes = el("details");
    changes.append(el("summary", "", `Promene (${r.changes.length})`), changeList(r.changes, "Bez promena u mapi i listama."));
    const files = el("details");
    files.append(el("summary", "", `Fajlovi (${r.files.length})`));
    const ul = el("ul");
    for (const f of r.files) {
      const li = el("li");
      const a = el("a", "", f.path);
      a.href = `/api/catalog/releases/${r.number}/files/${f.path.split("/").map(encodeURIComponent).join("/")}`;
      a.target = "_blank";
      li.append(a);
      ul.append(li);
    }
    files.append(ul);
    card.append(changes, files);
    if (!r.live) {
      const back = el("button", "btn small", "Vrati na ovo izdanje");
      back.onclick = () => askReason(card, "Zašto se vraćaš na ovo izdanje?", async (reason) => {
        await postJson(`/api/catalog/releases/${r.number}/rollback`, { actor: person(), role: roleSelect.value, reason });
      });
      card.append(back);
    }
    return card;
  }

  function lockBlock(lock) {
    const box = el("section", "catalogCardSection");
    box.append(el("h3", "", "Zaključavanje Mira i Worda"));
    if (lock.locked) {
      box.append(el("div", "catalogSubhead", `Zaključano ${formatTime(lock.since)} (izdanje ${lock.release}). Katalog je jedini izvor, Miro i Word su arhiva.`));
      box.append(el("div", "catalogSubhead", "Tekst koji sam upisuješ na vrh Mira i Worda:"));
      box.append(el("div", "catalogNoDesc", lock.archive_text));
    } else {
      box.append(el("div", "catalogSubhead", "Dugme radi tek kad su ispunjena sva tri uslova. Do tada se Miro i Word vode kao i do sada."));
      box.append(checkList(lock.conditions.map((c) => ({ ok: c.ok, label: `${c.label}. ${c.detail}`, problems: [] }))));
    }
    const form = el("div", "catalogAsk");
    const note = el("input", "catalogReason");
    note.placeholder = lock.locked ? "Zašto se vraćaš na Miro i Word?" : "Napomena";
    const button = el("button", lock.locked ? "btn" : "btn primary", lock.locked ? "Otključaj (povratak)" : "Zaključaj");
    const msg = el("span", "catalogPasteMsg");
    button.disabled = !lock.locked && !lock.can_lock;
    button.onclick = async () => {
      button.disabled = true;
      try {
        await postJson("/api/catalog/lock", { action: lock.locked ? "unlock" : "lock", actor: person(), role: roleSelect.value, reason: note.value.trim() });
        showToast(lock.locked ? "Otključano." : "Zaključano.");
        loadClasses();
      } catch (e) { msg.textContent = e.message; button.disabled = false; }
    };
    form.append(note, button, msg);
    box.append(form);
    return box;
  }

  function renderReleases(next, list, lock) {
    const nodes = [lockBlock(lock), el("h2", "catalogTitle", "Sledeće izdanje ")];
    nodes[1].append(el("span", "catalogCount", `poredi se sa: ${next.baseline_label}`));
    nodes.push(el("div", "catalogSubhead",
      "Rad u senci: objavljeno izdanje ide u poseban folder i pipeline ga još ne čita. Mapa, liste i dokumenti u produkciji se ne menjaju."));
    nodes.push(checkList(next.checks));
    nodes.push(el("h3", "catalogVarHead", "Šta se menja"));
    nodes.push(changeList(next.changes, next.same_as_live ? "Isto kao važeće izdanje." : "Nema promena."));
    nodes.push(parityBlock(next.parity));
    const form = el("div", "catalogAsk");
    const note = el("input", "catalogReason");
    note.placeholder = "Napomena: šta ovo izdanje donosi";
    const publish = el("button", "btn primary", "Objavi izdanje");
    const msg = el("span", "catalogPasteMsg");
    publish.disabled = !next.can_publish;
    if (!next.can_publish) msg.textContent = next.same_as_live ? "Nema šta da se objavi." : "Objava nije moguća dok provere ne prođu.";
    publish.onclick = async () => {
      publish.disabled = true;
      try {
        const r = await postJson("/api/catalog/releases", { actor: person(), role: roleSelect.value, reason: note.value.trim() });
        showToast(`Izdanje ${r.number} je objavljeno.`);
        loadClasses();
      } catch (e) { msg.textContent = e.message; publish.disabled = false; }
    };
    form.append(note, publish, msg);
    nodes.push(form);
    nodes.push(el("h2", "catalogTitle", "Objavljena izdanja"));
    if (!list.releases.length) nodes.push(el("div", "catalogHint small", "Još nijedno izdanje nije objavljeno."));
    list.releases.forEach((r) => nodes.push(releaseRow(r)));
    body.replaceChildren(...nodes);
  }

  // ---------------- Model cards (read only) ----------------

  function shown(v, fmt) {
    if (v === null || v === undefined || v === "") return NOT_RECORDED;
    if (typeof v === "boolean") return v ? "da" : "ne";
    return fmt ? fmt(v) : String(v);
  }

  function score(v) { return Number(v).toLocaleString("sr-RS", { minimumFractionDigits: 3, maximumFractionDigits: 3 }); }

  function factRows(rows) {
    const table = el("table", "catalogFactTable");
    for (const [label, item, fmt] of rows) {
      const tr = el("tr");
      tr.append(el("th", "", label));
      const td = el("td", item && item.value !== null && item.value !== undefined ? "" : "catalogMissing", shown(item && item.value, fmt));
      if (item && item.source) td.title = `Izvor: ${item.source}`;
      tr.append(td);
      table.append(tr);
    }
    return table;
  }

  function cardSection(title, source, ...content) {
    const box = el("section", "catalogCardSection");
    box.append(el("h3", "", title));
    if (source) box.append(el("div", "catalogSource", `Izvor: ${source}`));
    box.append(...content);
    return box;
  }

  function balanceSection(card) {
    const nodes = [];
    const cfg = card.balance_config;
    if (cfg) {
      const auto = cfg.auto || {};
      const lines = [
        ["Najmanje primera po klasi", { value: cfg.min_samples_per_class === "auto" ? "automatski" : cfg.min_samples_per_class }],
        ["Cilj (percentil broja primera)", { value: auto.percentile !== undefined ? `P${auto.percentile}` : null }],
        ["Cilj najmanje / najviše", { value: auto.min_target !== undefined ? `${auto.min_target} / ${auto.max_target}` : null }],
        ["Najviše dodatih po klasi", { value: cfg.max_aug_per_class }],
        ["Isecanje na deo kamere (ROI)", { value: cfg.roi_crop }],
        ["Izmene na isečku", { value: cfg.roi_transforms && cfg.roi_transforms.length ? cfg.roi_transforms.join(", ") : null }],
        ["Izmene na celoj slici", { value: cfg.global_transforms && cfg.global_transforms.length ? cfg.global_transforms.join(", ") : null }],
      ];
      const box = el("div");
      box.append(el("div", "catalogSource", `Izvor: ${cfg.source}. Fajlovi: ${cfg.files.join(", ")}.`));
      if (cfg.changed_after_training.length) {
        box.append(el("div", "catalogNoDesc",
          "Pažnja: posle ovog treninga menjano je: " + cfg.changed_after_training.map((c) => `${c.file} (${formatDate(c.changed)})`).join(", ")
          + ". Vrednosti možda nisu iste kao u treningu."));
      }
      box.append(factRows(lines));
      if (cfg.adaptive_strategy) {
        const steps = cfg.adaptive_strategy.map((s) => s.threshold >= 1e9 ? `ostale ${Math.round(s.percent * 100)}%`
          : `do ${s.threshold} primera ${Math.round(s.percent * 100)}%`);
        box.append(el("div", "catalogSubhead", `Koliko se slika dodaje po veličini klase: ${steps.join(", ")}.`));
      }
      nodes.push(box);
    } else {
      nodes.push(el("div", "catalogMissing", `Podešavanje balansiranja: ${NOT_RECORDED}.`));
    }
    const plan = card.balance_plan;
    if (plan) {
      nodes.push(el("div", "catalogSubhead",
        `Stvarno u treningu (izvor: ${plan.source}): ${shown(plan.real_images)} pravih slika, ${shown(plan.added_images)} dodatih.`));
      const table = el("table", "catalogTable");
      const head = el("tr");
      for (const t of ["Klasa", "Dodato", "Train", "Val", "Eval"]) head.append(el("th", t === "Klasa" ? "" : "num", t));
      table.append(head);
      for (const r of plan.classes) {
        const tr = el("tr");
        tr.append(el("td", "", r.class));
        for (const k of ["added", "train", "val", "eval"]) tr.append(el("td", "num", shown(r[k])));
        table.append(tr);
      }
      nodes.push(table);
    } else {
      nodes.push(el("div", "catalogMissing", `Stvarni plan po klasi: ${NOT_RECORDED} (arhiva trening seta ne postoji).`));
    }
    const section = cardSection("Dodatno balansiranje dataseta", null, ...nodes);
    section.classList.add("wide");
    return section;
  }

  function renderCard(card) {
    const t = card.training;
    const wrap = el("div", "catalogModelCard");
    if (card.problems.length) for (const p of card.problems) wrap.append(el("div", "catalogNoDesc", p));
    wrap.append(cardSection("Osnovno", null, factRows([
      ["Verzija", card.version], ["Datum treninga", card.date, formatDate], ["Arhitektura", card.architecture],
      ["Početni model", card.base_model], ["Ultralytics", card.ultralytics], ["Fajl", { value: card.file }],
    ])));
    wrap.append(cardSection("Trening", null, factRows([
      ["Epohe (planirano)", t.epochs], ["Odrađeno epoha", t.epochs_run], ["Najbolja epoha", t.best_epoch],
      ["Patience", t.patience], ["Veličina slike", t.imgsz], ["Batch", t.batch], ["Optimizer", t.optimizer],
      ["Početni lr", t.lr0], ["Završni lr (deo)", t.lrf], ["Kosinusni lr", t.cos_lr], ["Seed", t.seed],
      ["Bez mozaika poslednjih epoha", t.close_mosaic],
    ])));
    const augLabels = { hsv_h: "Nijansa (hsv_h)", hsv_s: "Zasićenje (hsv_s)", hsv_v: "Svetlina (hsv_v)", degrees: "Rotacija",
      translate: "Pomeranje", scale: "Razmera", shear: "Iskošenje", perspective: "Perspektiva", flipud: "Okret gore-dole",
      fliplr: "Okret levo-desno", mosaic: "Mozaik", mixup: "Mixup", cutmix: "Cutmix", copy_paste: "Copy-paste",
      auto_augment: "Auto augment", erasing: "Brisanje dela" };
    wrap.append(cardSection("Augmentacije tokom treninga", null,
      factRows(Object.keys(augLabels).map((k) => [augLabels[k], card.augmentations[k]]))));
    wrap.append(balanceSection(card));
    const names = el("details", "catalogGroup");
    names.append(el("summary", "", `Klase (${shown(card.classes.count)})`));
    names.append(card.classes.names.length ? el("div", "catalogSubhead", card.classes.names.join(", ")) : el("div", "catalogMissing", NOT_RECORDED));
    wrap.append(cardSection("Podaci", card.classes.source ? `klase: ${card.classes.source}` : null, factRows([
      ["Slika za trening", card.samples.train], ["Slika za proveru (val)", card.samples.val],
    ]), names));
    wrap.append(cardSection("Rezultati (val, najbolja epoha)", null, factRows([
      ["Precision", card.results.precision, score], ["Recall", card.results.recall, score],
      ["mAP50", card.results.mAP50, score], ["mAP50-95", card.results["mAP50-95"], score],
    ])));
    const promo = card.promotions.length
      ? factRows(card.promotions.map((p) => [`Promovisan ${formatDate(p.date)}`,
        { value: [p.before && p.before !== "(nije zabeleženo)" ? `umesto ${p.before}` : null, p.why].filter(Boolean).join(". ") || "razlog nije zabeležen", source: p.source }]))
      : el("div", "catalogMissing", `Promocija: ${NOT_RECORDED}.`);
    wrap.append(cardSection("Promocija", null, promo));
    return wrap;
  }

  function versionsTable(entry, onPick) {
    const table = el("table", "catalogTable selectable");
    const head = el("tr");
    for (const title of ["Verzija", "Datum", "Arhitektura", "Klase", "Train / Val", "mAP50", "Epohe"]) head.append(el("th", "", title));
    table.append(head);
    for (const c of entry.cards) {
      const tr = el("tr", c.file === openCardFile ? "selected" : "");
      tr.title = c.file;
      tr.append(el("td", "", `${shown(c.version.value)}${c.live ? " (živi)" : ""}`), el("td", "", shown(c.date.value, formatDate)),
        el("td", "", shown(c.architecture.value)), el("td", "num", shown(c.classes.count)),
        el("td", "num", `${shown(c.samples.train.value)} / ${shown(c.samples.val.value)}`),
        el("td", "num", shown(c.results.mAP50.value, score)),
        el("td", "num", `${shown(c.training.epochs_run.value)} od ${shown(c.training.epochs.value)}`));
      tr.onclick = () => onPick(c);
      table.append(tr);
    }
    return table;
  }

  function renderModelCards(model, data) {
    if (!data.snapshot) {
      showMessage("Kartice modela se upravo prave. Stranica će se sama osvežiti za par sekundi.");
      return;
    }
    const entry = data.snapshot.models.find((m) => m.model.toLowerCase() === model.toLowerCase());
    if (!entry) { showMessage(`Za model ${model} nema fajlova modela na serveru.`); return; }
    const card = entry.cards.find((c) => c.file === openCardFile) || entry.cards.find((c) => c.live) || entry.cards[0];
    const title = el("h2", "catalogTitle", `${model}: ${card.live ? "živi model" : "prethodna verzija"} `);
    title.append(el("span", "catalogCount", `${shown(card.version.value)}, ${shown(card.date.value, formatDate)}`));
    const nodes = [title, el("div", "catalogSubhead",
      `Samo za pregled. Puni se sama iz fajlova modela, evidencije i trening konfiguracije. Pređi mišem preko vrednosti da vidiš izvor. Osveženo ${formatTime(data.snapshot.built_at)}.`)];
    if (data.refreshing) nodes.push(el("div", "catalogSubhead", "Fajlovi su se promenili, kartice se osvežavaju..."));
    for (const w of entry.warnings) nodes.push(el("div", "catalogNoDesc", w));
    if (!card.live) {
      const back = el("button", "btn small", "Nazad na živi model");
      back.onclick = () => { openCardFile = null; renderModelCards(model, data); };
      nodes.push(back);
    }
    nodes.push(renderCard(card));
    nodes.push(el("h3", "catalogVarHead", "Sve verzije"));
    nodes.push(versionsTable(entry, (c) => { openCardFile = c.file; renderModelCards(model, data); body.scrollTop = 0; }));
    body.replaceChildren(...nodes);
  }

  async function loadClasses(keepFocus) {
    const no = ++requestNo;
    openArticle = null;
    try {
      if (view === "articles") {
        const params = new URLSearchParams();
        if (articleQuery.trim()) params.set("q", articleQuery.trim());
        if (articleFilter) params.set("state", articleFilter);
        const data = await getJson(`/api/catalog/articles?${params}`);
        if (no === requestNo) renderArticles(data);
        return;
      }
      if (view === "decisions") {
        const data = await getJson(`/api/catalog/decisions${decisionFilter ? `?kind=${decisionFilter}` : ""}`);
        if (no === requestNo) renderDecisions(data);
        return;
      }
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
      if (view === "inbox") {
        const data = await getJson("/api/catalog/inbox");
        if (no === requestNo) await renderInbox(data);
        return;
      }
      if (view === "days") {
        const data = await getJson("/api/catalog/day-table");
        if (no === requestNo) renderDays(data);
        return;
      }
      if (view === "findings") {
        const data = await getJson(`/api/catalog/findings?status=${findingFilter}`);
        if (no === requestNo) renderFindings(data);
        return;
      }
      if (view === "releases") {
        const [next, list, lock] = await Promise.all([getJson("/api/catalog/releases/next"), getJson("/api/catalog/releases"),
          getJson("/api/catalog/lock")]);
        if (no === requestNo) renderReleases(next, list, lock);
        return;
      }
      if (view === "modelcards") {
        const model = modelSelect.value;
        const data = await getJson("/api/catalog/model-cards");
        if (no !== requestNo) return;
        renderModelCards(model, data);
        if (data.refreshing && cardRetries < 15) {
          cardRetries += 1;
          setTimeout(() => { if (no === requestNo && view === "modelcards") loadClasses(); }, 2000);
        } else if (!data.refreshing) {
          cardRetries = 0;
        }
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
    viewArticles.classList.toggle("primary", view === "articles");
    viewDecisions.classList.toggle("primary", view === "decisions");
    viewModelCards.classList.toggle("primary", view === "modelcards");
    viewReleases.classList.toggle("primary", view === "releases");
    viewFindings.classList.toggle("primary", view === "findings");
    viewDays.classList.toggle("primary", view === "days");
    viewInbox.classList.toggle("primary", view === "inbox");
    selectedVariation = null;
    openCardFile = null;
    cardRetries = 0;
    modelSelect.closest("label").classList.toggle("hidden", !["classes", "miro", "noimage", "modelcards"].includes(view));
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
  fetch("/api/catalog/status").then((r) => { tab.hidden = !r.ok; tabInbox.hidden = !r.ok; }).catch(() => {});

  roleSelect.value = readRole();
  function applyRole() {
    const admin = roleSelect.value === "admin";
    viewReport.classList.toggle("hidden", !admin);
    viewMiro.classList.toggle("hidden", !admin);
    viewEvidence.classList.toggle("hidden", !admin);
    viewDecisions.classList.toggle("hidden", !admin);
    viewModelCards.classList.toggle("hidden", !admin);
    viewReleases.classList.toggle("hidden", !admin);
    viewFindings.classList.toggle("hidden", !admin);
    if (!admin && ["report", "miro", "evidence", "decisions", "modelcards", "releases", "findings"].includes(view)) setView("classes");
    else if (openArticle !== null && !panel.classList.contains("hidden")) showArticle(openArticle);
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
  viewArticles.onclick = () => setView("articles");
  viewDecisions.classList.toggle("hidden", roleSelect.value !== "admin");
  viewDecisions.onclick = () => setView("decisions");
  viewReport.onclick = () => setView("report");
  viewModelCards.classList.toggle("hidden", roleSelect.value !== "admin");
  viewModelCards.onclick = () => setView("modelcards");
  viewReleases.classList.toggle("hidden", roleSelect.value !== "admin");
  viewReleases.onclick = () => setView("releases");
  viewFindings.classList.toggle("hidden", roleSelect.value !== "admin");
  viewFindings.onclick = () => setView("findings");
  viewDays.onclick = () => setView("days");
  viewInbox.onclick = () => setView("inbox");
  modelSelect.onchange = () => { selectedClass = null; selectedVariation = null; openCardFile = null; loadClasses(); };
  viewClasses.onclick = () => setView("classes");
  viewUnclassified.onclick = () => setView("unclassified");
  tab.onclick = () => { setOpen(true); refresh(); };
  tabInbox.onclick = () => { setOpen(true); tab.classList.remove("tabActive"); tabInbox.classList.add("tabActive"); setView("inbox"); };
  tabs.addEventListener("click", (e) => {
    const button = e.target.closest(".tabBtn");
    if (button && button !== tab && button !== tabInbox) { setOpen(false); tabInbox.classList.remove("tabActive"); }
  }, true);
  // The labeling shortcuts listen on window: keep them from acting on the hidden workspace.
  for (const type of ["keydown", "keyup"]) {
    window.addEventListener(type, (e) => {
      if (!panel.classList.contains("hidden")) e.stopPropagation();
    }, true);
  }
})();
