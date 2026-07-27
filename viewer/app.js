// FIRE Index Viewer - Google Search Style
(function () {
  let currentKey = null;
  let currentTab = "all";

  // ── Elements ──
  const $ = id => document.getElementById(id);
  const landingPage = $("landingPage");
  const resultsPage = $("resultsPage");
  const landingSearch = $("landingSearch");
  const resultsSearch = $("resultsSearch");
  const landingClear = $("landingClear");
  const resultsClear = $("resultsClear");
  const indexSelect = $("indexSelect");
  const indexSelectResults = $("indexSelectResults");
  const resultsContainer = $("resultsContainer");
  const resultStats = $("resultStats");

  function init() {
    // Populate index selects
    Object.keys(INDEX_DATA).forEach((k, i) => {
      const name = INDEX_DATA[k].name;
      indexSelect.innerHTML += `<option value="${k}" ${i===0?'selected':''}>${name}</option>`;
      indexSelectResults.innerHTML += `<option value="${k}" ${i===0?'selected':''}>${name}</option>`;
    });
    currentKey = Object.keys(INDEX_DATA)[0];

    // Events
    $("btnSearchLanding").addEventListener("click", () => doSearch());
    $("btnBrowse").addEventListener("click", () => doSearch("*"));
    $("btnSearchResults").addEventListener("click", () => doSearch());
    landingSearch.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });
    resultsSearch.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });
    landingSearch.addEventListener("input", () => toggleClear(landingSearch, landingClear));
    resultsSearch.addEventListener("input", () => toggleClear(resultsSearch, resultsClear));
    landingClear.addEventListener("click", () => { landingSearch.value = ""; landingSearch.focus(); toggleClear(landingSearch, landingClear); });
    resultsClear.addEventListener("click", () => { resultsSearch.value = ""; resultsSearch.focus(); toggleClear(resultsSearch, resultsClear); });
    $("backToLanding").addEventListener("click", showLanding);
    indexSelect.addEventListener("change", () => { currentKey = indexSelect.value; indexSelectResults.value = currentKey; });
    indexSelectResults.addEventListener("change", () => { currentKey = indexSelectResults.value; indexSelect.value = currentKey; doSearch(); });

    // Theme toggles
    [$("themeToggle"), $("themeToggle2")].forEach(btn => btn.addEventListener("click", toggleTheme));

    // Tabs
    document.querySelectorAll(".tab").forEach(t => {
      t.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
        t.classList.add("active");
        currentTab = t.dataset.tab;
        renderResults(resultsSearch.value.trim());
      });
    });

    // Load saved theme
    if (localStorage.getItem("fire-theme") === "dark") applyTheme("dark");
    landingSearch.focus();
  }

  function toggleClear(input, btn) {
    btn.classList.toggle("hidden", !input.value);
  }

  function toggleTheme() {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
    localStorage.setItem("fire-theme", next);
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme === "dark" ? "dark" : "");
    const isDark = theme === "dark";
    [$("sunIcon"), $("sunIcon2")].forEach(el => el.classList.toggle("hidden", isDark));
    [$("moonIcon"), $("moonIcon2")].forEach(el => el.classList.toggle("hidden", !isDark));
  }

  function showLanding() {
    resultsPage.classList.add("hidden");
    landingPage.classList.remove("hidden");
    landingSearch.value = resultsSearch.value;
    landingSearch.focus();
  }

  function doSearch(override) {
    const query = override || (landingPage.classList.contains("hidden") ? resultsSearch.value.trim() : landingSearch.value.trim());
    if (!query) return;
    currentKey = indexSelect.value;
    indexSelectResults.value = currentKey;

    landingPage.classList.add("hidden");
    resultsPage.classList.remove("hidden");
    resultsSearch.value = query === "*" ? "" : query;
    toggleClear(resultsSearch, resultsClear);

    // Reset to All tab
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    $("tabAll").classList.add("active");
    currentTab = "all";

    renderResults(query === "*" ? "" : query);
  }

  // ── Rendering ──
  function renderResults(query) {
    const idx = INDEX_DATA[currentKey];
    const inv = idx.inverted_index;
    const docs = idx.documents;
    const filter = query.toLowerCase();

    if (currentTab === "config") {
      renderConfig(idx);
      return;
    }

    let html = "";
    const t0 = performance.now();

    if (currentTab === "all" || currentTab === "terms") {
      html += renderTermResults(inv, docs, filter);
    }
    if (currentTab === "all" || currentTab === "docs") {
      html += renderDocResults(docs, inv, filter);
    }

    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    const termCount = Object.keys(inv).filter(t => !filter || t.includes(filter)).length;
    resultStats.textContent = filter
      ? `About ${termCount} results (${elapsed} seconds) for "${filter}" in ${idx.name}`
      : `Showing all ${Object.keys(inv).length} terms and ${Object.keys(docs).length} documents from ${idx.name}`;
    resultsContainer.innerHTML = html;

    // Attach click handlers for expandable terms
    resultsContainer.querySelectorAll(".result-title[data-term]").forEach(el => {
      el.addEventListener("click", () => expandTerm(el.dataset.term));
    });
    resultsContainer.querySelectorAll(".result-title[data-did]").forEach(el => {
      el.addEventListener("click", () => expandDoc(el.dataset.did));
    });
  }

  function renderTermResults(inv, docs, filter) {
    let entries = Object.entries(inv);
    if (filter) entries = entries.filter(([t]) => t.includes(filter));
    entries.sort((a, b) => Object.keys(b[1]).length - Object.keys(a[1]).length);

    const show = entries.slice(0, 50);
    let html = "";
    show.forEach(([term, postings]) => {
      const df = Object.keys(postings).length;
      const docNames = Object.keys(postings).slice(0, 4).map(did => {
        const d = docs[did];
        return d ? d.title : `Doc #${did}`;
      });
      const snippet = `Appears in <strong>${df}</strong> document${df !== 1 ? "s" : ""}. Found in: ${docNames.join(", ")}${df > 4 ? ", ..." : ""}`;

      const chips = Object.entries(postings).slice(0, 6).map(([did, val]) => {
        const v = Array.isArray(val) ? `${val.length} pos` : val;
        return `<span class="chip">Doc #${did}: ${v}</span>`;
      }).join("");

      html += `
        <div class="result">
          <div class="result-url-row">
            <div class="result-favicon">T</div>
            <div class="result-url-text">
              <span class="result-site">${INDEX_DATA[currentKey].name}</span>
              <span class="result-path">inverted_index / ${term}</span>
            </div>
          </div>
          <a class="result-title" data-term="${term}">${term}</a>
          <div class="result-snippet">${snippet}</div>
          <div class="result-chips">${chips}</div>
        </div>`;
    });

    if (entries.length > 50) {
      html += `<div style="color:var(--text-secondary);font-size:13px;margin:16px 0;">Showing 50 of ${entries.length} matching terms. Refine your search to see more.</div>`;
    }
    return html;
  }

  function renderDocResults(docs, inv, filter) {
    let docEntries = Object.entries(docs);
    if (filter) {
      docEntries = docEntries.filter(([, d]) => d.title.toLowerCase().includes(filter));
    }

    // Count terms per doc
    const termCounts = {};
    for (const postings of Object.values(inv)) {
      for (const did of Object.keys(postings)) {
        termCounts[did] = (termCounts[did] || 0) + 1;
      }
    }

    let html = "";
    docEntries.forEach(([did, doc]) => {
      const tc = termCounts[did] || 0;
      html += `
        <div class="result">
          <div class="result-url-row">
            <div class="result-favicon">D</div>
            <div class="result-url-text">
              <span class="result-site">${INDEX_DATA[currentKey].name}</span>
              <span class="result-path">documents / doc_${did}</span>
            </div>
          </div>
          <a class="result-title" data-did="${did}">${doc.title || "Untitled Document"}</a>
          <div class="result-snippet">Published: <strong>${doc.date || "N/A"}</strong> &mdash; ${tc} unique terms indexed in this document.</div>
        </div>`;
    });
    return html;
  }

  function renderConfig(idx) {
    const labels = { x: "Information Indexed", y: "Datastore", z: "Compression", i: "Index Optimization", q: "Query Engine" };
    let configRows = Object.entries(idx.config).map(([k, v]) =>
      `<div class="config-row"><span class="config-key">${labels[k] || k}</span><span class="config-val">${v}</span></div>`
    ).join("");

    resultStats.textContent = `Index configuration for ${idx.name}`;
    resultsContainer.innerHTML = `
      <div class="config-card">
        <h3>${idx.name}</h3>
        <p style="color:var(--text-secondary);font-size:14px;margin-bottom:12px;">${idx.description}</p>
        ${configRows}
        <div class="config-row" style="margin-top:8px;border-top:1px solid var(--border);padding-top:8px;">
          <span class="config-key">Total Terms</span><span class="config-val">${idx.num_terms}</span>
        </div>
        <div class="config-row">
          <span class="config-key">Total Documents</span><span class="config-val">${idx.num_docs}</span>
        </div>
      </div>`;
  }

  // ── Expand a term inline ──
  function expandTerm(term) {
    const existing = document.querySelector(`.detail-panel[data-for="${term}"]`);
    if (existing) { existing.remove(); return; }

    const postings = INDEX_DATA[currentKey].inverted_index[term];
    const docs = INDEX_DATA[currentKey].documents;
    let rows = Object.entries(postings).map(([did, val]) => {
      const title = docs[did] ? docs[did].title : "Unknown";
      const valStr = Array.isArray(val) ? `Positions: [${val.join(", ")}]` : val;
      return `<tr><td>${did}</td><td>${title}</td><td>${valStr}</td></tr>`;
    }).join("");

    const panel = document.createElement("div");
    panel.className = "detail-panel";
    panel.setAttribute("data-for", term);
    panel.innerHTML = `
      <h3>"${term}" &mdash; ${Object.keys(postings).length} document(s)</h3>
      <table class="detail-table">
        <thead><tr><th>Doc ID</th><th>Title</th><th>Value</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;

    // Insert after the clicked result
    const resultEl = document.querySelector(`.result-title[data-term="${term}"]`).closest(".result");
    resultEl.after(panel);
  }

  function expandDoc(did) {
    const existing = document.querySelector(`.detail-panel[data-for="doc-${did}"]`);
    if (existing) { existing.remove(); return; }

    const inv = INDEX_DATA[currentKey].inverted_index;
    const doc = INDEX_DATA[currentKey].documents[did];
    const terms = [];
    for (const [term, postings] of Object.entries(inv)) {
      if (postings[did] !== undefined) {
        const val = postings[did];
        terms.push({ term, val: Array.isArray(val) ? `${val.length} positions` : val });
      }
    }
    terms.sort((a, b) => a.term.localeCompare(b.term));

    let rows = terms.map(t => `<tr><td>${t.term}</td><td>${t.val}</td></tr>`).join("");

    const panel = document.createElement("div");
    panel.className = "detail-panel";
    panel.setAttribute("data-for", `doc-${did}`);
    panel.innerHTML = `
      <h3>${doc.title} &mdash; ${terms.length} terms</h3>
      <p style="color:var(--text-secondary);font-size:13px;margin-bottom:10px;">Date: ${doc.date || "N/A"}</p>
      <table class="detail-table">
        <thead><tr><th>Term</th><th>Value</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;

    const resultEl = document.querySelector(`.result-title[data-did="${did}"]`).closest(".result");
    resultEl.after(panel);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
