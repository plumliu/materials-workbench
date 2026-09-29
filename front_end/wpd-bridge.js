// Same-origin adapter to the pinned WPD build. Image pixels stay in their original frame.
(() => {
  let ready = false;
  const snapshot = () => JSON.stringify(wpd.appData.getPlotData().serialize(wpd.appData.getFileManager().getMetadata()));
  wpd.handleLaunchArgs = () => { ready = true; };
  window.onbeforeunload = null; // The outer editor owns the actual unsaved-change check.
  function resetEditor() {
    for (const popup of document.querySelectorAll('.popup')) {
      if (popup.style.visibility === 'visible') wpd.popup.close(popup.id);
    }
    wpd.sidebar.clear();
    wpd.toolbar.clear();
    wpd.graphicsWidget.removeTool();
    wpd.graphicsWidget.removeRepainter();
    wpd.appData.reset();
    wpd.appData.setPageManager(null);
  }
  window.workbench = {
    get ready() { return ready; },
    snapshot,
    async loadPDF(blob, name) {
      const file = new File([blob], name, {type: 'application/pdf'});
      resetEditor();
      wpd.imageManager.initializeFileManager([file], true);
      await wpd.imageManager.loadFromFile(file, true);
      wpd.tree.refresh();
      wpd.tree.selectPath('/' + wpd.gettext('axes'));
      wpd.graphicsWidget.zoomFit();
      return snapshot();
    },
    async load(blob) {
      const reader = new tarball.TarReader();
      const files = await reader.readFile(blob);
      const infos = files.filter(f => f.name.endsWith('/info.json'));
      if (infos.length !== 1) throw Error('TAR 缺少唯一的 info.json');
      const root = infos[0].name.slice(0, -9);
      const info = JSON.parse(reader.getTextFile(infos[0].name));
      const project = JSON.parse(reader.getTextFile(root + info.json));
      const images = info.images.map(name => {
        const type = name.toLowerCase().endsWith('.pdf') ? 'application/pdf' : 'image/png';
        return new File([reader.getFileBlob(root + name, type)], name, {type});
      });
      resetEditor();
      wpd.imageManager.initializeFileManager(images, true);
      await wpd.imageManager.loadFromFile(images[0], true);
      const metadata = wpd.appData.getPlotData().deserialize(project);
      wpd.appData.getFileManager().loadMetadata(metadata || {});
      wpd.graphicsWidget.resetData();
      wpd.graphicsWidget.removeTool();
      wpd.graphicsWidget.removeRepainter();
      wpd.tree.refresh();
      wpd.tree.selectPath('/' + wpd.gettext(wpd.appData.getPlotData().getDatasetCount() ? 'datasets' : 'axes'));
      wpd.graphicsWidget.zoomFit();
      return snapshot();
    },
    async archive(json) {
      const images = Array.from(wpd.appData.getFileManager().getFiles());
      const writer = new tarball.TarWriter();
      writer.addFolder('chart/');
      writer.addTextFile('chart/info.json', JSON.stringify({version:[4,0],json:'wpd.json',images:images.map(file => file.name)}));
      writer.addTextFile('chart/wpd.json', json);
      for (const file of images) writer.addFile('chart/' + file.name, file);
      return writer.writeBlob();
    }
  };
})();
