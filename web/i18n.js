/* UI-only translations: user names and server-provided content are never translated. */
(() => {
  const en = {
    "Темнее синий — сильнее осадки. Капли — условная анимация; дождь и снег не разделены.": "Darker blue means heavier precipitation. Droplets are symbolic; rain and snow are not distinguished.",
    "Редкий дождь → плотные осадки": "Light → heavy precipitation",
    "Тучи и дождь — условное изображение осадков. Плотность зависит от интенсивности, не от облачности.": "Clouds and rain symbolize precipitation. Density reflects precipitation intensity, not cloud cover.",
    "В центре карты": "Map center",
    "Облачность пока недоступна у подключённого источника.": "Cloud cover is not currently available from the connected source.",
    "Нет погодных данных для видимой области. Попробуйте обновить слой.": "No weather data for this view. Try refreshing the layer.",
    "Погодные данные покрывают не всю видимую область. Нельзя подтвердить отсутствие осадков.": "Weather data does not cover the entire view. Absence of precipitation cannot be confirmed.",
    "Цветные зоны — осадки, мм/ч. Это не облачность; дождь и снег здесь не разделены.": "Colored areas show precipitation in mm/h, not clouds. Rain and snow are not distinguished.",
    "В видимой области на выбранный час осадки не прогнозируются. Это не означает отсутствие облаков.": "No precipitation is predicted in this view at the selected time. Clouds may still be present.",
    "Изотермы через 2 °C. Подписи показывают температуру воздуха.": "Isotherms every 2 °C. Labels show air temperature.",
    "Здесь температура почти одинаковая. Отдалите карту, чтобы увидеть изотермы.": "Temperature is nearly uniform here. Zoom out to see isotherms.",
    "Погода": "Weather",
    "Погода на карте": "Weather map",
    "Закрыть погоду": "Close weather",
    "Погодный слой": "Weather layer",
    "Температура": "Temperature",
    "Осадки": "Precipitation",
    "Время прогноза": "Forecast time",
    "Непрозрачность": "Opacity",
    "Повторить загрузку": "Retry",
    "мм/ч": "mm/h",
    "Прогноз GFS · сетка около 28 км. Не измерения на участке.": "GFS forecast · approx. 28 km grid. Not plot measurements.",
    "Нет актуальных погодных данных.": "No current weather data available.",
    "Загрузка погодного слоя…": "Loading weather layer…",
    "Погода недоступна: не настроен ключ MapTiler.": "Weather unavailable: MapTiler key is not configured.",
    "Не удалось загрузить погоду. Проверьте интернет или лимит MapTiler.": "Weather failed to load. Check your connection or MapTiler quota.",
    "Часть погоды не загрузилась. Проверьте интернет или лимит MapTiler.": "Some weather tiles failed to load. Check your connection or MapTiler quota.",
    "Прозрачные области — без осадков по модели.": "Transparent areas: no precipitation predicted by the model.",
    "Цвет показывает температуру воздуха на высоте 2 м.": "Color shows air temperature 2 m above ground.",
    "Об участке": "Plot details",
    "Контур участка": "Plot boundary",
    "Анализ участка": "Plot analysis",
    "Выберите период наблюдений для оценки состояния растительности.": "Choose an observation period to assess vegetation health.",
    "Информация об участке": "Plot information",
    "Общая информация": "Overview",
    "Графики": "Charts",
    "Графики появятся после анализа": "Charts will appear after analysis",
    "Выберите период и запустите анализ на вкладке «Общая информация».": "Choose a period and run analysis in the Overview tab.",
    "К общей информации": "Go to overview",
    "Спутник": "Satellite",
    "MapTiler · фотоснимки": "MapTiler · imagery",
    "Спутник: снимки разных дат, не в реальном времени. Детализация зависит от региона.": "Satellite: imagery from different dates, not live. Detail varies by region.",
    "Спутник недоступен: проверьте локальную настройку ключа MapTiler.": "Satellite unavailable: check the local MapTiler key configuration.",
    "Слои карты": "Map layers",
    "Выберите оформление подложки. Ваши участки останутся на карте.": "Choose a basemap style. Your plots stay on the map.",
    "Стандартный": "Standard",
    "Светлый": "Light",
    "Тёмный": "Dark",
    "Топографический": "Topographic",
    "OSM · светлая палитра": "OSM · light palette",
    "OSM · тёмная палитра": "OSM · dark palette",
    "Объекты на карте": "Map overlays",
    "Часть карты не загрузилась. Попробуйте другой слой или проверьте интернет.": "Some map tiles failed to load. Try another layer or check your connection.",
    "Закрыть уведомление": "Dismiss notification",
    "Приблизить": "Zoom in",
    "Отдалить": "Zoom out",
    "Моё местоположение": "My location",
    "Геолокация недоступна в этом браузере. Используйте HTTPS или localhost.": "Geolocation is unavailable. Use HTTPS or localhost.",
    "Доступ к местоположению запрещён. Разрешите его в настройках браузера.": "Location access denied. Allow it in your browser settings.",
    "Не удалось определить местоположение вовремя. Попробуйте ещё раз.": "Location request timed out. Please try again.",
    "Местоположение недоступно. Проверьте настройки геолокации устройства.": "Location unavailable. Check your device location settings.",
    "Настройки": "Settings",
    "Настройки карты": "Map settings",
    "Закрыть настройки": "Close settings",
    "Настройте интерфейс и карту под себя.": "Make the interface and map work for you.",
    "Интерфейс": "Interface",
    "Язык интерфейса": "Interface language",
    "Применяется сразу, без перезагрузки": "Applied instantly, no reload needed",
    "Карта": "Map",
    "Рабочее пространство": "Workspace",
    "ВИД КАРТЫ": "MAP STYLE",
    "Схема": "Standard",
    "Спутник · скоро": "Satellite · coming soon",
    "Спутниковый слой пока не подключён": "Satellite imagery is not connected yet",
    "Контуры OSM": "OSM boundaries",
    "Сохранённые участки видны всегда": "Saved plots are always visible",
    "Вернуть панель инструментов на место": "Reset toolbar position",
    "Настройки сохраняются в этом браузере.": "Settings are saved in this browser.",
    "Город, страна или сохранённый участок": "City, country or saved plot",
    "Город, страна или участок": "City, country or plot",
    "Найти": "Search",
    "Результаты поиска": "Search results",
    "Сохранённые поля": "Saved plots",
    "Карта полей": "Plot map",
    "Загрузка карты…": "Loading map…",
    "Инструменты выделения": "Selection tools",
    "Переместить панель инструментов": "Move toolbar",
    "Перетащите панель · стрелки — перемещение · Home или двойной клик — сброс": "Drag toolbar · arrow keys to move · Home or double-click to reset",
    "Перемещение карты · сбросить выбор поля": "Pan map · clear plot selection",
    "Перемещение карты": "Pan map",
    "Курсор · выбор и перемещение карты": "Cursor · select and pan",
    "Курсор": "Cursor",
    "Прямоугольник · два противоположных угла": "Rectangle · two opposite corners",
    "Прямоугольник": "Rectangle",
    "Круг · центр и радиус": "Circle · center and radius",
    "Круг": "Circle",
    "Полигон · построение по точкам": "Polygon · draw point by point",
    "Полигон": "Polygon",
    "Свободный контур · зажмите кнопку мыши и обведите поле": "Freehand · hold and trace a plot",
    "Свободный контур": "Freehand",
    "Найти участок · выберите точку на карте": "Find a plot · select a point on the map",
    "Найти участок": "Find a plot",
    "Мои поля": "My plots",
    "КАТАЛОГ УЧАСТКОВ": "PLOT CATALOG",
    "Участки": "Plots",
    "Закрыть сохранённые поля": "Close saved plots",
    "СОХРАНЁННЫЕ КОНТУРЫ": "SAVED BOUNDARIES",
    "Конкурсные временные ряды": "Competition time series",
    "У этих данных нет координат. Анализ доступен отдельно от карты.": "These data have no coordinates. Analysis is available separately from the map.",
    "Полигон датасета": "Dataset plot",
    "Выберите ряд": "Select a series",
    "Карта · OpenStreetMap / ODbL": "Map · OpenStreetMap / ODbL",
    "Выбранное поле": "Selected plot",
    "ВЫБРАННОЕ ПОЛЕ": "SELECTED PLOT",
    "Закрыть карточку": "Close plot details",
    "Площадь": "Area",
    "Культура": "Crop",
    "Неизвестна": "Unknown",
    "Название поля": "Plot name",
    "Например, Северное поле": "For example, North field",
    "Добавить в мои поля": "Add to my plots",
    "Сохранить название": "Save name",
    "Сезон анализа": "Analysis season",
    "Начало периода": "Start date",
    "Конец периода": "End date",
    "Sentinel-2 + ERA5-Land. История поля загружается автоматически.": "Sentinel-2 + ERA5-Land. Plot history is loaded automatically.",
    "Проанализировать": "Analyze",
    "Анализируем…": "Analyzing…",
    "Создаём задание…": "Creating job…",
    "Повторить анализ": "Retry analysis",
    "Проверить обновления": "Check for updates",
    "Проверить анализ": "Check analysis",
    "Сбор спутниковых данных": "Collecting satellite data",
    "Удалить из моих полей": "Delete from my plots",
    "Редактор контура": "Boundary editor",
    "Отметьте вершины поля": "Mark the plot vertices",
    "Замкните границу нажатием на первую точку.": "Click the first point to close the boundary.",
    "↶ Назад": "↶ Undo",
    "Вершины": "Vertices",
    "Фигура": "Shape",
    "Замкнуть": "Close boundary",
    "Отмена": "Cancel",
    "Сохранить поле": "Save plot",
    "Нажмите на контур поля, чтобы открыть его карточку": "Click a plot boundary to open its details",
    "Анализ NDVI": "NDVI analysis",
    "ДИНАМИКА ВЕГЕТАЦИИ": "VEGETATION TRENDS",
    "NDVI и сезонная норма": "NDVI and seasonal baseline",
    "Наблюдения": "Observations",
    "Восстановленный ряд": "Reconstructed series",
    "Норма ± 1σ": "Baseline ± 1σ",
    "Отклонения": "Anomalies",
    "Наведите на график для подробностей": "Hover over the chart for details",
    "Полоса — разброс исторических значений, не уверенность прогноза. Возможные причины отклонений требуют проверки.": "The band shows historical variation, not prediction confidence. Potential causes of anomalies require verification.",
    "Сохранённые участки": "Saved plots",
    "Города и страны": "Cities and countries",
    "Город / регион / страна": "City / region / country",
    "Панель инструментов возвращена на исходное место": "Toolbar position reset"
  };
  const reverse = new Map(Object.entries(en).map(([ru, english]) => [english, ru]));
  let language = "ru";
  const bindings = [];
  const normalize = text => text.trim().replace(/\s+/g, " ");
  const translate = text => {
    const key = normalize(text);
    const ru = reverse.get(key) || key;
    return language === "en" ? en[ru] || text : (reverse.get(key) || text);
  };
  function init() {
    try { language = localStorage.getItem("agropulse.language") === "en" ? "en" : "ru"; } catch {}
    // Capture only initial, authored UI text. Do not bind user/data containers.
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (node.parentElement.closest("script, #fieldTitle, #fieldList, #searchResults, #polygonSelect, #chartTitle")) continue;
      if (en[normalize(node.data)]) bindings.push({ node, original: node.data });
    }
    document.querySelectorAll("[title], [aria-label], [placeholder]").forEach(node => {
      for (const attribute of ["title", "aria-label", "placeholder"]) {
        const original = node.getAttribute(attribute);
        if (original && en[original]) bindings.push({ node, attribute, original });
      }
    });
    // These containers contain UI copy only, never plot names or backend results.
    const dynamic = ["saveFieldButton", "analyzeButton", "verticesButton", "drawTitle", "drawHint", "toastMessage", "basemapStatus"];
    const refreshDynamic = () => dynamic.forEach(id => {
      const node = document.getElementById(id);
      const result = translate(node.textContent);
      if (node.textContent !== result) node.textContent = result;
    });
    const apply = () => {
      document.documentElement.lang = language;
      document.title = language === "en" ? "AgroPulse — plot map" : "АгроПульс — карта полей";
      bindings.forEach(({ node, attribute, original }) => {
        if (attribute) node.setAttribute(attribute, translate(original));
        else if (node.isConnected) node.data = translate(original);
      });
      refreshDynamic();
      document.dispatchEvent(new Event("agropulse:languagechange"));
    };
    const observer = new MutationObserver(refreshDynamic);
    dynamic.forEach(id => observer.observe(document.getElementById(id), { childList: true, characterData: true, subtree: true }));
    const select = document.getElementById("languageSelect");
    select.value = language;
    select.onchange = () => {
      language = select.value === "en" ? "en" : "ru";
      try { localStorage.setItem("agropulse.language", language); } catch {}
      apply();
    };
    apply();
  }
  window.AgroI18n = { init, translate };
})();
