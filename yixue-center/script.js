const i18n = {
  zh: {
    brand: "易学与世界文明研究中心",
    nav_about: "About",
    nav_events: "Events",
    nav_publications: "Publications",
    nav_fellowship: "Fellowship",
    nav_contact: "Contact",
    hero_eyebrow: "上海海洋大学外国语学院",
    hero_title: "面向世界文明语境的《周易》研究平台",
    hero_sub: "成立于2026年1月，聚焦《周易》文本考辨、经典互证与跨文明对话，推动易学研究的纵向溯源与横向互鉴。",
    hero_cta: "了解中心",
    about_title: "About",
    about_p1: "易学与世界文明研究中心是依托上海海洋大学外国语学院成立的学术机构，专门从事《周易》研究与跨文明阐释。",
    about_p2: "本中心以《周易》为核心研究对象，立足文本考辨，贯通中国经典体系，并面向世界文明语境，推进易学研究的纵向溯源与横向互鉴。",
    dir1_title: "《周易》的文本研究",
    dir1_desc: "系统比勘传世文本与出土文献（如马王堆帛书、阜阳汉简、上博楚简等），厘清文本形成与演变脉络。",
    dir2_title: "《周易》与其他经典互参",
    dir2_desc: "与《尚书》《诗经》《黄帝内经》等进行思想互证，探讨《周易》在中华文明谱系中的枢纽地位。",
    dir3_title: "《周易》与世界文明对话",
    dir3_desc: "开展与自然法传统及荣格分析心理学的比较研究，关注海外译介与诠释，促进文明互鉴。",
    events_title: "Events",
    event1_title: "“《周易》与经典传统”青年论坛",
    event1_desc: "面向青年学者征稿，聚焦文本、思想与方法的跨学科讨论。",
    event2_title: "国际研讨会：Yijing and World Civilizations",
    event2_desc: "邀请海内外学者围绕译介史、比较哲学与文明互鉴展开对话。",
    pub_title: "Publications",
    pub1: "《周易》文本系统比勘导论（工作论文）",
    pub2: "Yijing and Natural Law Traditions: A Comparative Framework",
    pub3: "《周易》海外译介史料目录（第一辑）",
    fellow_title: "Fellowship",
    fellow_p: "中心设访问学者与青年研究员项目，欢迎历史文献学、哲学、宗教学、翻译学、比较文明等领域学者申请。",
    fellow_cta: "咨询申请方式",
    contact_title: "Contact",
    contact_p: "欢迎学术合作、讲座邀请与咨询合作。请通过以下方式与我们联系。",
    contact_email_label: "邮箱：",
    contact_addr_label: "地址：",
    contact_addr: "上海市临港新城沪城环路999号，上海海洋大学外国语学院",
    form_name: "姓名 / Name",
    form_email: "邮箱 / Email",
    form_msg: "留言 / Message",
    form_submit: "发送 / Submit",
    footer: "© 2026 易学与世界文明研究中心 版权所有。"
  },
  en: {
    brand: "Center for Yijing and World Civilizations",
    nav_about: "About",
    nav_events: "Events",
    nav_publications: "Publications",
    nav_fellowship: "Fellowship",
    nav_contact: "Contact",
    hero_eyebrow: "Shanghai Ocean University · College of Foreign Languages",
    hero_title: "A Global Platform for Yijing Studies and Civilizational Dialogue",
    hero_sub: "Founded in January 2026, the center advances philological Yijing studies, classical cross-reference, and intercultural scholarship.",
    hero_cta: "Discover the Center",
    about_title: "About",
    about_p1: "The Center for Yijing and World Civilizations is an academic institute affiliated with the College of Foreign Languages at Shanghai Ocean University.",
    about_p2: "Taking the Zhouyi as its core object, the center combines textual criticism, classical integration, and global civilizational perspectives for both historical depth and comparative breadth.",
    dir1_title: "Textual Studies of the Zhouyi",
    dir1_desc: "We collate received texts with excavated manuscripts (e.g., Mawangdui silk manuscripts, Fuyang Han slips, and Shanghai Museum Chu slips) to clarify textual formation and transformation.",
    dir2_title: "Cross-Reading with Chinese Classics",
    dir2_desc: "Through dialogue with the Shangshu, Shijing, and Huangdi Neijing, we examine the Zhouyi as a structural hub in the Chinese intellectual tradition.",
    dir3_title: "Dialogue with World Civilizations",
    dir3_desc: "We develop comparative work with natural law traditions and Jungian analytical psychology while engaging global translations and interpretations of the Yijing.",
    events_title: "Events",
    event1_title: "Young Scholars Forum: Zhouyi and Classical Traditions",
    event1_desc: "Call for papers focused on textual inquiry, thought history, and interdisciplinary methods.",
    event2_title: "International Symposium: Yijing and World Civilizations",
    event2_desc: "Scholars worldwide discuss translation history, comparative philosophy, and civilizational exchange.",
    pub_title: "Publications",
    pub1: "Introduction to Systematic Collation of Zhouyi Texts (Working Paper)",
    pub2: "Yijing and Natural Law Traditions: A Comparative Framework",
    pub3: "Bibliography of Global Yijing Translation Histories (Series I)",
    fellow_title: "Fellowship",
    fellow_p: "The center offers visiting and junior fellowship tracks for researchers in philology, philosophy, religious studies, translation, and comparative civilizations.",
    fellow_cta: "Inquire About Applications",
    contact_title: "Contact",
    contact_p: "For academic collaboration, lectures, and consulting opportunities, please contact us.",
    contact_email_label: "Email:",
    contact_addr_label: "Address:",
    contact_addr: "College of Foreign Languages, Shanghai Ocean University, 999 Huchenghuan Road, Lingang, Shanghai",
    form_name: "Name",
    form_email: "Email",
    form_msg: "Message",
    form_submit: "Submit",
    footer: "© 2026 Center for Yijing and World Civilizations. All rights reserved."
  }
};

let currentLang = "zh";
const langBtn = document.getElementById("langToggle");

function render(lang) {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (i18n[lang][key]) el.textContent = i18n[lang][key];
  });
  langBtn.textContent = lang === "zh" ? "EN" : "中文";
}

langBtn.addEventListener("click", () => {
  currentLang = currentLang === "zh" ? "en" : "zh";
  render(currentLang);
});

render(currentLang);
