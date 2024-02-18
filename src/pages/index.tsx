import React from 'react';

import Head from 'next/head';

import Canvas from '../components/Canvas';
import Contact from '../components/Email';
import Header from '../components/Header';
import Introduction from '../components/Intro';
import LazyShow from '../components/LazyShow';
import Product from '../components/Product';
import Upload_image from '../components/Upload_image';

const App = () => {
  return (
    <div className={`bg-background grid gap-y-16 overflow-hidden`}>
      <Head>
        <title>
          Free Online Image to LaTeX Converter
        </title>
        <meta
          name="description"
          content="Convert images of LaTeX Math Equations / Matrices online for free with AI powered Latex-OCR. Edit LaTeX formula in-app and paste the result directly into your document."
          key="desc"
        />
      </Head>
      <LazyShow>
        <>
          <Header />
          <Product />
          <Introduction />
          <Upload_image />
        </>
      </LazyShow>
      <LazyShow>
        <>
          <Contact />
          <Canvas />
        </>
      </LazyShow>
    </div>
  );
};

export default App;
